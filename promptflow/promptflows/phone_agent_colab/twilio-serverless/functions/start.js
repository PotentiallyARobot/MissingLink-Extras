const crypto = require('crypto');

exports.handler = async function (context, event, callback) {
  const {
    baseUrl,
    jsonResponse,
    unauthorizedJson,
    checkAdmin,
    safeInt,
    nowIso,
    createDocument,
    updateDocument,
  } = require(Runtime.getFunctions()['private/lib'].path);

  try {
    if (!checkAdmin(context, event)) {
      return callback(null, unauthorizedJson());
    }

    const to = String(event.to || '').trim();
    const objective = String(event.objective || '').trim();
    const contactName = String(event.contact_name || '').trim();
    const maxTurns = Math.max(2, Math.min(12, safeInt(event.max_turns, safeInt(context.DEFAULT_MAX_TURNS, 6))));

    if (!to || !objective) {
      return callback(null, jsonResponse({ ok: false, error: 'Missing to or objective' }, 400));
    }

    const client = context.getTwilioClient();
    const jobId = `job_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
    const adminToken = crypto.randomBytes(16).toString('hex');

    const state = {
      job_id: jobId,
      admin_token: adminToken,
      to,
      objective,
      contact_name: contactName,
      status: 'queued',
      result: '',
      confidence: 'low',
      error: '',
      call_sid: '',
      turn_count: 0,
      max_turns: maxTurns,
      turns: [],
      created_at: nowIso(),
      updated_at: nowIso(),
      ended_at: null,
    };

    await createDocument(context, jobId, state);

    const call = await client.calls.create({
      to,
      from_: context.TWILIO_FROM_NUMBER,
      url: `${baseUrl(context)}/voice?job_id=${encodeURIComponent(jobId)}&admin_token=${encodeURIComponent(adminToken)}`,
      method: 'POST',
      statusCallback: `${baseUrl(context)}/status-callback?job_id=${encodeURIComponent(jobId)}&admin_token=${encodeURIComponent(adminToken)}`,
      statusCallbackMethod: 'POST',
      statusCallbackEvent: ['initiated', 'ringing', 'answered', 'completed'],
    });

    state.call_sid = call.sid;
    state.status = 'initiated';
    state.updated_at = nowIso();
    await updateDocument(context, jobId, state);

    return callback(
      null,
      jsonResponse({
        ok: true,
        job_id: jobId,
        admin_token: adminToken,
        call_sid: call.sid,
        status: state.status,
        poll_url: `${baseUrl(context)}/status?job_id=${encodeURIComponent(jobId)}&admin_token=${encodeURIComponent(adminToken)}`,
      })
    );
  } catch (err) {
    return callback(null, jsonResponse({ ok: false, error: String(err.message || err) }, 500));
  }
};
