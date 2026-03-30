exports.handler = async function (context, event, callback) {
  const {
    fetchDocument,
    updateDocument,
    generateAgentAction,
    buildTwiml,
    nowIso,
  } = require(Runtime.getFunctions()['private/lib'].path);

  try {
    const jobId = String(event.job_id || '').trim();
    const adminToken = String(event.admin_token || '').trim();
    const doc = await fetchDocument(context, jobId);

    if (!doc || !doc.data || doc.data.admin_token !== adminToken) {
      const VoiceResponse = require('twilio').twiml.VoiceResponse;
      const response = new VoiceResponse();
      response.say('Sorry, this call session is not available. Goodbye.');
      response.hangup();
      return callback(null, response);
    }

    const state = doc.data;
    const speechResult = String(event.SpeechResult || '').trim();
    const digits = String(event.Digits || '').trim();
    let latestInput = '';

    if (digits) {
      latestInput = `[DTMF:${digits}]`;
    } else if (speechResult) {
      latestInput = speechResult;
    } else {
      latestInput = '[NO_INPUT]';
    }

    state.turns = state.turns || [];
    state.turns.push({ role: 'user', text: latestInput, at: nowIso() });
    state.updated_at = nowIso();
    state.status = 'in_progress';

    const action = await generateAgentAction(context, state, latestInput);
    state.turns.push({ role: 'assistant', text: action.say, at: nowIso() });
    state.turn_count = (state.turn_count || 0) + 1;

    if (action.result) {
      state.result = action.result;
      state.confidence = action.confidence || state.confidence || 'low';
    }

    const maxTurns = parseInt(state.max_turns || context.DEFAULT_MAX_TURNS || '6', 10);
    if ((state.turn_count || 0) >= maxTurns && !(action.done || action.hangup)) {
      action.say = 'Thank you. Goodbye.';
      action.done = true;
      action.hangup = true;
      if (!state.result) {
        state.result = 'Call ended after reaching the maximum turn limit without a final confirmed answer.';
        state.confidence = 'low';
      }
    }

    if (action.hangup || action.done) {
      state.status = 'done';
      state.ended_at = nowIso();
    }

    await updateDocument(context, jobId, state);
    const twiml = buildTwiml(context, state, jobId, adminToken, action);
    return callback(null, twiml);
  } catch (err) {
    const VoiceResponse = require('twilio').twiml.VoiceResponse;
    const response = new VoiceResponse();
    response.say('Sorry, an error occurred. Goodbye.');
    response.hangup();
    return callback(null, response);
  }
};
