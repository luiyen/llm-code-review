import * as core from '@actions/core'
import github from '@actions/github'
import {generate} from './generate'
import {minimatch} from 'minimatch'
import parseDiff, {Chunk, File} from 'parse-diff'

interface ReviewResponse {
  lineNumber: string
  reviewComment: string
}

function createPrompt(file: File, chunk: Chunk): string {
  return `Your task is to review pull requests. Instructions:
- Provide the response in following JSON format:  [{"lineNumber":  <line_number>, "reviewComment": "<review comment>"}]
- Do not give positive comments or compliments.
- Provide comments and suggestions ONLY if there is something to improve, otherwise return an empty array.
- Write the comment in GitHub Markdown format.
- Use the given description only for the overall context and only comment the code.
- IMPORTANT: NEVER suggest adding comments to the code.

Review the following code diff in the file "${file.to}".
  
Git diff to review:

\`\`\`diff
${chunk.content}
${chunk.changes
  // @ts-expect-error - ln and ln2 exists where needed
  .map(c => `${c.ln ? c.ln : c.ln2} ${c.content}`)
  .join('\n')}
\`\`\`
`
}

function leaveComment(
  file: File,
  reviews: ReviewResponse[]
): {body: string; path: string; line: number}[] {
  return reviews.flatMap(review => {
    if (!file.to) {
      return []
    }
    return {
      body: review.reviewComment,
      path: file.to,
      line: Number(review.lineNumber)
    }
  })
}
async function reviewCode(
  parsedDiff: File[],
  endpoint_url: string,
  api_key: string,
  max_new_tokens: string,
  temperature: string,
  top_k: string,
  top_p: string
): Promise<{body: string; path: string; line: number}[]> {
  const comments: {body: string; path: string; line: number}[] = []

  for (const file of parsedDiff) {
    if (file.to === '/dev/null') continue // Ignore deleted files
    for (const chunk of file.chunks) {
      const prompt = createPrompt(file, chunk)
      const response = await generate(
        prompt,
        endpoint_url,
        api_key,
        max_new_tokens,
        temperature,
        top_k,
        top_p
      )
      const review: ReviewResponse[] | undefined =
        (JSON.parse(response) as ReviewResponse[]) || undefined
      if (review) {
        const newComments = leaveComment(file, review)
        if (newComments) {
          comments.push(...newComments)
        }
      }
    }
  }
  return comments
}

async function run(): Promise<void> {
  try {
    const githubToken = core.getInput('githubToken', {required: true})
    const prNumber = github.context.payload.pull_request?.number || 0
    const repo = github.context.repo
    const octokit = github.getOctokit(githubToken)
    const diff = await octokit.rest.pulls.get({
      owner: repo.owner,
      repo: repo.repo,
      pull_number: prNumber,
      mediaType: {format: 'diff'}
    })
    const parsedDiff = parseDiff(String(diff.data))

    const excludePatterns = core
      .getInput('exclude')
      .split(',')
      .map(s => s.trim())

    const filteredDiff = parsedDiff.filter(file => {
      return !excludePatterns.some(pattern => minimatch(file.to ?? '', pattern))
    })

    const endpoint_url = core.getInput('url')
    const api_key = core.getInput('apiKey')
    const max_new_tokens = core.getInput('maxNewTokens')
    const temperature = core.getInput('temperature')
    const top_k = core.getInput('topK')
    const top_p = core.getInput('topP')

    reviewCode(
      filteredDiff,
      endpoint_url,
      api_key,
      max_new_tokens,
      temperature,
      top_k,
      top_p
    )
    core.setOutput('diff', parsedDiff)
  } catch (error) {
    if (error instanceof Error) core.setFailed(error.message)
  }
}

run()
