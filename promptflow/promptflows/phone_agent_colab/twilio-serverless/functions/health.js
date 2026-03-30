exports.handler = async function (context, event, callback) {
  const { jsonResponse } = require(Runtime.getFunctions()['private/lib'].path);
  return callback(null, jsonResponse({ ok: true, service: 'colab-phone-agent' }));
};
