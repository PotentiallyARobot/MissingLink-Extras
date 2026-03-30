exports.handler = async function (context, event, callback) {
  const {
    fetchDocument,
    jsonResponse,
  } = require(Runtime.getFunctions()['private/lib'].path);

  try {
    const jobId = String(event.job_id || '').trim();
    const adminToken = String(event.admin_token || '').trim();
    const doc = await fetchDocument(context, jobId);

    if (!doc || !doc.data || doc.data.admin_token !== adminToken) {
      return callback(null, jsonResponse({ ok: false, error: 'Not found' }, 404));
    }

    const state = doc.data;
    return callback(null, jsonResponse({
      ok: true,
      job_id: state.job_id,
      status: state.status,
      to: state.to,
      objective: state.objective,
      result: state.result,
      confidence: state.confidence,
      call_sid: state.call_sid,
      turn_count: state.turn_count,
      max_turns: state.max_turns,
      created_at: state.created_at,
      updated_at: state.updated_at,
      ended_at: state.ended_at,
      turns: state.turns,
      error: state.error || '',
    }));
  } catch (err) {
    return callback(null, jsonResponse({ ok: false, error: String(err.message || err) }, 500));
  }
};
