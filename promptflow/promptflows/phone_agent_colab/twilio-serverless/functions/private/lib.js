const OpenAI = require('openai');

function nowIso() {
  return new Date().toISOString();
}

function baseUrl(context) {
  return `https://${context.DOMAIN_NAME}`;
}

function jsonResponse(body, statusCode = 200) {
  const response = new Twilio.Response();
  response.appendHeader('Content-Type', 'application/json');
  response.setStatusCode(statusCode);
  response.setBody(body);
  return response;
}

function unauthorizedJson() {
  return jsonResponse({ ok: false, error: 'Unauthorized' }, 401);
}

function checkAdmin(context, event) {
  return event.admin_key === context.APP_ADMIN_KEY || event.admin_token === context.APP_ADMIN_KEY;
}

function safeInt(value, fallback) {
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? n : fallback;
}

async function fetchDocument(context, jobId) {
  const client = context.getTwilioClient();
  try {
    return await client.sync.v1
      .services(context.SYNC_SERVICE_SID)
      .documents(jobId)
      .fetch();
  } catch (err) {
    if ((err.status && err.status === 404) || String(err.message || '').includes('20404')) {
      return null;
    }
    throw err;
  }
}

async function createDocument(context, jobId, data) {
  const client = context.getTwilioClient();
  return client.sync.v1.services(context.SYNC_SERVICE_SID).documents.create({
    uniqueName: jobId,
    data,
    ttl: 60 * 60 * 24,
  });
}

async function updateDocument(context, jobId, data) {
  const client = context.getTwilioClient();
  return client.sync.v1.services(context.SYNC_SERVICE_SID).documents(jobId).update({
    data,
    ttl: 60 * 60 * 24,
  });
}

function trimTranscript(turns = [], maxTurns = 12) {
  return turns.slice(-maxTurns).map((turn) => ({
    role: turn.role,
    text: String(turn.text || '').trim(),
    at: turn.at || nowIso(),
  }));
}

function extractFirstJson(text) {
  const raw = String(text || '').trim();
  if (!raw) {
    throw new Error('Empty model output');
  }
  const noFence = raw
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/```$/i, '')
    .trim();

  const firstBrace = noFence.indexOf('{');
  const lastBrace = noFence.lastIndexOf('}');
  const candidate = firstBrace >= 0 && lastBrace > firstBrace
    ? noFence.slice(firstBrace, lastBrace + 1)
    : noFence;

  return JSON.parse(candidate);
}

function normalizeAction(parsed, fallbackSay) {
  return {
    say: String(parsed.say || fallbackSay || 'Sorry, could you repeat that?').trim(),
    listen_prompt: String(parsed.listen_prompt || '').trim(),
    done: Boolean(parsed.done),
    hangup: Boolean(parsed.hangup || parsed.done),
    result: String(parsed.result || '').trim(),
    confidence: ['low', 'medium', 'high'].includes(parsed.confidence) ? parsed.confidence : 'low',
    send_digits: String(parsed.send_digits || '').trim(),
    wait_before_digits_seconds: Math.max(0, Math.min(10, safeInt(parsed.wait_before_digits_seconds, 0))),
  };
}

async function generateAgentAction(context, state, latestUserInput = '') {
  const openai = new OpenAI({ apiKey: context.OPENAI_API_KEY });
  const transcript = trimTranscript(state.turns || []);
  const model = context.OPENAI_MODEL || 'gpt-5.4-nano';
  const maxTurns = safeInt(state.max_turns, safeInt(context.DEFAULT_MAX_TURNS, 6));

  const developerPrompt = [
    'You are a phone-call assistant running an outbound task.',
    'You MUST identify yourself as an automated assistant on your first spoken turn.',
    'Keep spoken replies short, natural, and easy to understand over a phone line.',
    'Never use markdown.',
    'Your job is to pursue the objective, ask follow-up questions only when necessary, and end the call once the objective is complete.',
    'If you have the requested information, say thank you, goodbye, and end the call.',
    'If the person refuses, clearly says they do not know, or the task cannot be completed, end politely and summarize that outcome in result.',
    'If the other side asks for keypad input, you may set send_digits to a short DTMF string like 1 or 123#.',
    'Output ONLY valid JSON with keys: say, listen_prompt, done, hangup, result, confidence, send_digits, wait_before_digits_seconds.',
    'Do not include any keys besides those.',
    'say must be under 220 characters.',
    'listen_prompt should usually be empty unless you want a short extra prompt before listening again.',
  ].join(' ');

  const userPrompt = {
    objective: state.objective,
    contact_name: state.contact_name || '',
    latest_user_input: latestUserInput,
    turn_count: state.turn_count || 0,
    max_turns: maxTurns,
    transcript,
    guidance: [
      'If this is the first turn, greet briefly, disclose automation, and ask the most direct question needed to complete the objective.',
      'If the objective is already satisfied, set done=true, hangup=true, and provide result.',
      'If turn_count is at or above max_turns, end the call politely and summarize the best known result.',
      'If there was no input, gently ask once more or end if repeated silence makes progress unlikely.',
    ],
  };

  const response = await openai.responses.create({
    model,
    reasoning: { effort: 'low' },
    input: [
      { role: 'developer', content: developerPrompt },
      { role: 'user', content: JSON.stringify(userPrompt) },
    ],
  });

  const parsed = extractFirstJson(response.output_text || '');
  return normalizeAction(parsed, 'Sorry, could you repeat that?');
}

function buildTwiml(context, state, jobId, adminToken, action) {
  const VoiceResponse = require('twilio').twiml.VoiceResponse;
  const response = new VoiceResponse();

  if (action.say) {
    response.say({ language: context.DEFAULT_LANGUAGE || 'en-US' }, action.say);
  }

  if (action.send_digits) {
    if (action.wait_before_digits_seconds > 0) {
      response.pause({ length: action.wait_before_digits_seconds });
    }
    response.play({ digits: action.send_digits });
  }

  if (action.hangup || action.done) {
    response.hangup();
    return response;
  }

  const gather = response.gather({
    input: 'speech dtmf',
    action: `${baseUrl(context)}/handle?job_id=${encodeURIComponent(jobId)}&admin_token=${encodeURIComponent(adminToken)}`,
    method: 'POST',
    speechTimeout: 'auto',
    timeout: 5,
    actionOnEmptyResult: true,
  });

  if (action.listen_prompt) {
    gather.say({ language: context.DEFAULT_LANGUAGE || 'en-US' }, action.listen_prompt);
  }

  response.say({ language: context.DEFAULT_LANGUAGE || 'en-US' }, 'Goodbye.');
  response.hangup();
  return response;
}

module.exports = {
  nowIso,
  baseUrl,
  jsonResponse,
  unauthorizedJson,
  checkAdmin,
  safeInt,
  fetchDocument,
  createDocument,
  updateDocument,
  trimTranscript,
  generateAgentAction,
  buildTwiml,
};
