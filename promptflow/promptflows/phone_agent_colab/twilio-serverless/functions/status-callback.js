exports.handler = async function (context, event, callback) {
  const {
    fetchDocument,
    updateDocument,
    jsonResponse,
    nowIso,
  } = require(Runtime.getFunctions()['private/lib'].path);

  try {
    const jobId = String(event.job_id || '').trim();
    const adminToken = String(event.admin_token || '').trim();
    const callStatus = String(event.CallStatus || '').trim();
    const doc = await fetchDocument(context, jobId);

    if (!doc || !doc.data || doc.data.admin_token !== adminToken) {
      return callback(null, jsonResponse({ ok: false, error: 'Not found' }, 404));
    }

    const state = doc.data;
    state.status = callStatus || state.status;
    state.updated_at = nowIso();

    if (callStatus === 'completed' && !state.ended_at) {
      state.ended_at = nowIso();
      if (!state.result) {
        state.result = 'Call completed without a final extracted answer.';
        state.confidence = state.confidence || 'low';
      }
    }

    await updateDocument(context, jobId, state);
    return callback(null, jsonResponse({ ok: true }));
  } catch (err) {
    return callback(null, jsonResponse({ ok: false, error: String(err.message || err) }, 500));
  }
};
