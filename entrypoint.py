#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import os
from typing import List

import click
import requests
from loguru import logger

from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import transformers
from transformers import AutoTokenizer

def check_required_env_vars():
    """Check required environment variables"""
    required_env_vars = [
        "HUGGINGFACEHUB_API_TOKEN",
        "GH_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_PULL_REQUEST_NUMBER",
        "GIT_COMMIT_HASH",
    ]
    for required_env_var in required_env_vars:
        if os.getenv(required_env_var) is None:
            raise ValueError(f"{required_env_var} is not set")

def create_a_comment_to_pull_request(
        github_token: str,
        github_repository: str,
        pull_request_number: int,
        git_commit_hash: str,
        body: str):
    """Create a comment to a pull request"""
    headers = {
        "Accept": "application/vnd.github.v3.patch",
        "authorization": f"Bearer {github_token}"
    }
    data = {
        "body": body,
        "commit_id": git_commit_hash,
        "event": "COMMENT"
    }
    url = f"https://api.github.com/repos/{github_repository}/pulls/{pull_request_number}/reviews"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response

def chunk_string(input_string: str, chunk_size) -> List[str]:
    """Chunk a string"""

    chunked_inputs = []
    for i in range(0, len(input_string), chunk_size):
        chunked_inputs.append(input_string[i:i + chunk_size])
    return chunked_inputs

def get_review(
        repo_id: str,
        diff: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        prompt_chunk_size: int
):
    """Get a review"""
    # Chunk the prompt
    chunks = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)

    # There are likely more performant models
    # Please see https://www.sbert.net/docs/pretrained_models.html
    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    embedding_task = "feature-extraction"
    embeddings = HuggingFaceHubEmbeddings(
        model=embedding_model,
        task=embedding_task,
    )

    task = "text-generation"
    # Update this
    hugging_face_repo_id = repo_id

    endpoint = HuggingFaceEndpoint(
        repo_id=hugging_face_repo_id,
        task=task,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    tokenizer = AutoTokenizer.from_pretrained(hugging_face_repo_id)
    llm = ChatHuggingFace(
        llm=endpoint,
        tokenizer=tokenizer
    )

    summaries = []
    # This is the format required for Mistral and other LLMs
    # Please see: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "Hello"),
            ("assistant", "Hello, I am a helpful AI software analyst. I assist software developers in reviewing and writing source code for applications."),
            ("user", "Provide a very concise summary for the changes in a git diff generated from a pull request submitted by a developer on GitHub. Importantly, do not reference the usage of the `git diff` command or git commit hashes in the summary.\n\ngit diff: {diff}")
        ],
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    inputs = {
        'diff': diff
    }

    summary = chain.invoke(inputs)
    summaries.append(summary)

    reviews = []
    for changes in chunks:

        prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "Hello"),
                ("assistant", "Hello, I am a helpful AI software analyst. I assist software developers in reviewing and writing source code for applications."),
                ("user", "Summarize the following changes in a git diff generated from a pull request submitted by a developer on GitHub. Importantly, include the line number of the change in the summary.\n\ngit diff: {changes}")
            ],
        )

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        inputs = {
            'changes': changes
        }

        review = chain.invoke(inputs)
        reviews.append(review)

        # Prompt for suggested improvements
        prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "Hello"),
                ("assistant", "Hello, I am a helpful AI software analyst. I assist software developers in reviewing and writing source code for applications."),
                ("user", "Analyze the following changes in a git diff generated from a pull request submitted by a developer on GitHub. If you are able to, determine if these proposed changes might be improved upon in any manner, and recommend any of these improvements.\n\ngit diff: {changes}")
            ],
        )

        chain = (
            prompt
            | llm
            | StrOutputParser()
        )

        inputs = {
            'changes': changes
        }

        suggestion = chain.invoke(inputs)
        reviews.append(suggestion)

    return summaries, reviews

def format_review_comment(summaries: List[str], reviews: List[str]) -> str:
    """Format reviews"""

    joined_summaries = "\n".join(summaries)
    joined_reviews = "\n".join(reviews)

    comment = f"""<details>
    <summary>{joined_summaries}</summary>
    {joined_reviews}
    </details>
    """

    return comment

@click.command()
@click.option("--diff", type=click.STRING, required=True, help="File path to the diff generated for the pull request")
@click.option("--diff-chunk-size", type=click.INT, required=False, default=3500, help="Pull request diff")
@click.option("--repo-id", type=click.STRING, required=False, default="gpt-3.5-turbo", help="HuggingFace model repository ID")
@click.option("--temperature", type=click.FLOAT, required=False, default=0.1, help="Temperature")
@click.option("--max-new-tokens", type=click.INT, required=False, default=250, help="Max tokens")
@click.option("--top-p", type=click.FLOAT, required=False, default=1.0, help="Top N")
@click.option("--top-k", type=click.INT, required=False, default=1.0, help="Top T")
@click.option("--log-level", type=click.STRING, required=False, default="INFO", help="Logging level")
def main(
    diff: str,
    diff_chunk_size: int,
    repo_id: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    log_level: str
):

    # Set log level
    logger.level(log_level)

    # Check if necessary environment variables are set or not
    check_required_env_vars()

    # Open and read the contents from the file generated from `git diff`
    fh = open(diff)
    diff_content = fh.read()
    fh.close()
    logger.debug(f"git diff: {diff_content}")

    summaries, reviews = get_review(
        diff=diff_content,
        repo_id=repo_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        prompt_chunk_size=diff_chunk_size
    )
    logger.debug(f"Summarized review: {summaries}")
    logger.debug(f"Chunked reviews: {reviews}")

    # Format reviews
    review_comment = format_review_comment(summaries=summaries, reviews=reviews)

    # Create a comment to a pull request
    create_a_comment_to_pull_request(
        github_token=os.getenv("GH_TOKEN"),
        github_repository=os.getenv("GITHUB_REPOSITORY"),
        pull_request_number=int(os.getenv("GITHUB_PULL_REQUEST_NUMBER")),
        git_commit_hash=os.getenv("GIT_COMMIT_HASH"),
        body=review_comment
    )

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
