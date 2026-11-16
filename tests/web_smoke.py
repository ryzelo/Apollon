"""Exercise the local web API against a supported model.

Start apollon --serve, then run:
    python tests/web_smoke.py MODEL_PATH [http://127.0.0.1:8765]
The script creates real analysis runs in the server's normal local history.
"""
import json
import sys
import time
import http.client
import urllib.error
import urllib.parse

MODEL = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else 'http://127.0.0.1:8765'


# One kept-alive connection, like the browser uses. Results of several megabytes sent to a client that
# asks for "Connection: close" can be cut off by a reset on Windows.
URL = urllib.parse.urlsplit(BASE)
connection = http.client.HTTPConnection(URL.hostname, URL.port, timeout=30)


def request(path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    connection.request(method or ('GET' if data is None else 'POST'), path, body=data, headers={'Content-Type': 'application/json'})
    response = connection.getresponse()
    payload = response.read()
    if response.status >= 400:
        raise urllib.error.HTTPError(BASE + path, response.status, payload.decode(errors='replace'), response.headers, None)
    return json.loads(payload)


def submit(kind, component=None):
    payload = {'model': MODEL, 'prompt': 'Hello world', 'kind': kind, 'answer': 1}
    if component:
        payload['component'] = component
    return request('/api/runs', payload)


def finished(job, timeout=180):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = request('/api/runs/' + job['id'])
        if value['status'] not in ('queued', 'running', 'cancelling'):
            return value
        time.sleep(0.2)
    raise AssertionError('Timed out waiting for ' + job['id'])


# a server that was only just started may not be listening yet
for attempt in range(40):
    try:
        models = request('/api/models')
        break
    except OSError:
        connection.close()
        time.sleep(0.25)
else:
    raise SystemExit('No apollon server at ' + BASE)
assert isinstance(models['models'], list)
for kind, component in [('generate', None), ('decompose', None), ('probe', 'S0 MLP'), ('trace', 'S0 #0')]:
    job = finished(submit(kind, component))
    assert job['status'] == 'completed', job
    result = job['result']
    assert result['schema_version'] == 2
    assert result['tokens'] and result['output']
    assert 0 <= result['focus'] < len(result['tokens'])
    assert all(0 <= item['probability'] <= 1 for item in result['output'])
    if kind != 'generate':
        assert result['components'], result
    part = result['components'][0] if result['components'] else {}
    if kind == 'decompose':
        heads = [c for p in result['components'] for c in p['children'] if c['kind'] == 'head']
        assert heads and all(len(h['attention']) == len(result['tokens']) for h in heads), 'head attention per position'
        assert all(abs(sum(h['attention']) - 1) < 1e-3 for h in heads), 'attention rows sum to one'
        measured = [p for p in result['components'] if p.get('total') is not None and 'kl' in p]
        assert measured and all(p['output_without'] and 'raises' in p and 'lowers' in p for p in measured)
        assert all(p['kl'] >= 0 for p in measured)
    if kind == 'probe':
        assert len(part['positions']) == result['focus'] + 1
        assert part['groups'] and part['downstream'], part.keys()
        path = part['downstream'][0]
        assert path['layers'] and path['layers'][0]['layer'] == part['layer']
        assert all({'lens_base', 'lens_variant', 'components', 'neurons', 'heads'} <= set(x) for x in path['layers'])
        dosed = [u for u in part['children'] if 'dose' in u]
        assert dosed and all(len(u['dose']) == 6 for u in dosed)
        # removing a unit is the dose with factor 0: both are the same intervention
        for u in dosed:
            zero = next(d for d in u['dose'] if d['factor'] == 0)
            assert abs(zero['change'] + u['total']) < 1e-3, (u['label'], zero['change'], u['total'])
        assert result['rival']['text'] is not None
    if kind == 'trace':
        assert part['tree']['kind'] == 'neuron' and part['tree']['unit'] == part['unit']
        assert abs(part['tree']['activation'] - part['activation']) < 1e-4
        assert part['removals'] and part['scalars'] and part['groups']
        # the groups partition the activation exactly
        assert abs(sum(g['activation_contribution'] for g in part['groups']) - part['activation']) < 1e-3 * max(1, abs(part['activation']))
        for s in part['scalars']:
            assert abs(sum(x['value'] for x in s['top_p']) + s['rest_p'] - s['p']) < 1e-3 * max(1, abs(s['p']))
    print(kind, 'completed', len(result['components']), 'components', flush=True)

job = finished(submit('neuron', 'S1 #0'))
assert job['status'] == 'completed', job
neuron = job['result']['neuron']
assert job['result']['mode'] == 'neuron' and neuron['layer'] == 1 and neuron['unit'] == 0
assert neuron['inputs'] and neuron['outputs']['raises'] and neuron['outputs']['neurons']['count'] > 0
assert neuron['repertoire']['activation_min'] <= neuron['repertoire']['activation_max']
assert all(c['deviation'] < 1e-3 for c in job['result']['checks']), job['result']['checks']
print('Static neuron result passed', flush=True)

job = finished(submit('probe', 'S0 MLP; S0 Attn'))
assert job['status'] == 'completed', job
parts = job['result']['components']
assert len(parts) == 2 and len({part['id'] for part in parts}) == 2, parts
reports = request('/api/runs/' + job['id'] + '/artifacts')
assert len([name for name in reports if name.endswith('.html')]) == 2, reports
print('Multiple probe components and reports passed', flush=True)

job = submit('probe', 'S0 MLP')
request('/api/runs/' + job['id'] + '/cancel', {})
assert finished(job)['status'] == 'cancelled'
print('Cancellation passed', flush=True)
request('/api/runs/' + job['id'], method='DELETE')
assert all(item['id'] != job['id'] for item in request('/api/runs')['runs'])
try:
    request('/api/runs/' + job['id'])
except urllib.error.HTTPError as exc:
    assert exc.code == 404
else:
    raise AssertionError('Deleted run is still served')
print('Deletion passed', flush=True)
job = finished(request('/api/runs', {'model': MODEL + '/missing', 'prompt': 'Hello', 'kind': 'generate', 'answer': 1}))
assert job['status'] == 'failed' and job['error']
print('Invalid model failure passed', flush=True)
assert any(item['id'] == job['id'] for item in request('/api/runs')['runs'])
try:
    request('/api/runs', {'model': MODEL, 'prompt': '', 'kind': 'generate'})
except urllib.error.HTTPError as exc:
    assert exc.code == 400
else:
    raise AssertionError('Empty prompt was accepted')
print('Validation and history passed', flush=True)
