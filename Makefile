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

.PHONE: setup
setup:
	pip install --force-reinstall --no-cache pip==23.0.1 setuptools==67.6.1
	pip install --force-reinstall --no-cache -r requirements-dev.txt --use-deprecated=legacy-resolver

.PHONY: setup-dev
setup-dev:
	pip install --force-reinstall --no-cache pip==23.0.1 setuptools==67.6.1
	pip install --force-reinstall --no-cache -r requirements-dev.txt --use-deprecated=legacy-resolver
	pre-commit install

build-docker:
	docker build -t gpt-code-review-action .


lint:

lint-python:
	pylint --rcfile=.pylintrc gpt_code_review_action

lint-docker:
	hadolint Dockerfile
