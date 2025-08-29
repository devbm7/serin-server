"""Basic connection example.
"""

import redis

r = redis.Redis(
    host='redis-17316.c330.asia-south1-1.gce.redns.redis-cloud.com',
    port=17316,
    decode_responses=True,
    username="default",
    password="QVoPG0oMdB7i7L9TU7qfNB08vRRRrxKm",
)

success = r.set('foo', 'bar')
# True

result = r.get('foo')
print(result)
# >>> bar

