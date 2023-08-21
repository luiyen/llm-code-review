import './fetch-polyfill'
import {HfInference} from '@huggingface/inference'

const BASE_URL =
  'https://api-inference.huggingface.co/models/HuggingFaceH4/starchat-beta'

export async function generate(
  inputs: string,
  endpoint_url?: string,
  api_key?: string,
  max_new_tokens?: string,
  temperature?: string,
  top_k?: string,
  top_p?: string
): Promise<string> {
  return new Promise(async resolve => {
    const model = endpoint_url || BASE_URL

    const hgInference = new HfInference(api_key || '')
    // HFInference
    let response = ''
    for await (const output of hgInference.textGenerationStream({
      model,
      inputs,
      parameters: {
        max_new_tokens: parseInt(max_new_tokens || '1024'),
        temperature: parseFloat(temperature || '0.2'),
        top_k: parseInt(top_k || '50'),
        top_p: parseFloat(top_p || '0.9')
      }
    })) {
      if (output.token.text === '<|end|>') {
        break
      }
      response += output.token.text
    }
    resolve(response)
  })
}
