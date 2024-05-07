# llm-code-review-action
_This is a fork of the GitHub project originally authored by (https://github.com/luiyen/llm-code-review)[https://github.com/luiyen/llm-code-review]. Aside from some trivial modifications, all credit and praise is due to @luiyen._

A container GitHub Action to review a pull request by HuggingFace's LLM Model.

If the size of a pull request is over the maximum chunk size of the HuggingFace API, the Action will split the pull request into multiple chunks and generate review comments for each chunk.
And then the Action summarizes the review comments and posts a review comment to the pull request.

## Pre-requisites
We have to set a GitHub Actions secret `HUGGING_FACE_API_KEY` to use the HuggingFace API so that we securely pass it to the Action.

## Inputs

- `apiKey`: The HuggingFace API key to access the API.
- `githubToken`: The GitHub token to access the GitHub API.
- `githubRepository`: The GitHub repository to post a review comment.
- `githubPullRequestNumber`: The GitHub pull request number to post a review comment.
- `gitCommitHash`: The git commit hash to post a review comment.
- `pullRequestDiff`: The diff of the pull request to generate a review comment.
- `pullRequestDiffChunkSize`: The chunk size of the diff of the pull request to generate a review comment.
- `repoId`: LLM repository id on HuggingFace.
- `temperature`: The temperature to generate a review comment.
- `topP`: The top_p to generate a review comment.
- `topK`: The top_k to generate a review comment.
- `maxNewTokens`: The max_tokens to generate a review comment.
- `logLevel`: The log level to print logs.

As you might know, a model of HuggingFace has limitation of the maximum number of input tokens.
So we have to split the diff of a pull request into multiple chunks, if the size of the diff is over the limitation.
We can tune the chunk size based on the model we use.

## Example usage
Here is an example to use the Action to review a pull request of the repository.
The actual file is located at [`.github/workflows/test-action.yml`](.github/workflows/test-action.yml).


```yaml
name: "Test Code Review"

on:
  pull_request:
    paths-ignore:
      - "*.md"
      - "LICENSE"

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v3
      - name: "Get diff of the pull request"
        id: get_diff
        shell: bash
        env:
          PULL_REQUEST_HEAD_REF: "${{ github.event.pull_request.head.ref }}"
        run: |-
          git fetch origin "${{ env.PULL_REQUEST_HEAD_REF }}:${{ env.PULL_REQUEST_HEAD_REF }}"
          git checkout "${{ env.PULL_REQUEST_HEAD_REF }}"
          git diff "origin/${{ env.PULL_REQUEST_HEAD_REF }}" > "diff.txt"
          # shellcheck disable=SC2086
          echo "diff=$(cat "diff.txt")" >> $GITHUB_ENV
      - uses: jrgriffiniii/llm-code-review@v0.0.1
        name: "Code Review"
        id: review
        with:
          apiKey: ${{ secrets.API_KEY }}
          githubToken: ${{ secrets.GITHUB_TOKEN }}
          githubRepository: ${{ github.repository }}
          githubPullRequestNumber: ${{ github.event.pull_request.number }}
          gitCommitHash: ${{ github.event.pull_request.head.sha }}
          repoId: "meta-llama/Llama-2-7b-chat-hf"
          temperature: "0.2"
          maxNewTokens: "250"
          topK: "50"
          topP: "0.95"
          pullRequestDiff: |-
            ${{ steps.get_diff.outputs.pull_request_diff }}
          pullRequestChunkSize: "3500"
          logLevel: "DEBUG"
```
