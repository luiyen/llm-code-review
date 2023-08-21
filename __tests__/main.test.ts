import {generate} from '../src/generate'
import {expect, test} from '@jest/globals'

// shows how the runner will run a javascript action with env / stdout protocol
test.concurrent(
  'test runs',
  async () => {
    const result = await generate('hello', undefined, 'HUGGING_FACE_API_KEY')
    expect(result.length).toBeGreaterThan(0)
  },
  1000 * 20
)
