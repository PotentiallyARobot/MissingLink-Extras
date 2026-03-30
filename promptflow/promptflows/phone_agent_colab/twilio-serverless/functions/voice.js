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
      response.say('Sorry, this call could not be initialized. Goodbye.');
      response.hangup();
      return callback(null, response);
    }

    const state = doc.data;
    state.status = 'in_progress';
    state.updated_at = nowIso();

    const action = await generateAgentAction(context, state, '');
    state.turns = state.turns || [];
    state.turns.push({ role: 'assistant', text: action.say, at: nowIso() });
    state.turn_count = (state.turn_count || 0) + 1;

    if (action.result) {
      state.result = action.result;
      state.confidence = action.confidence || state.confidence || 'low';
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
