import fetch, {Headers, Request} from 'node-fetch-polyfill'
if (!globalThis.fetch) {
  globalThis.fetch = fetch
  globalThis.Headers = Headers
  globalThis.Request = Request
}
