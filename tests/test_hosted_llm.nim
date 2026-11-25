## Hosted calls must reach the native sidecar without provider credentials.
include "../src/escrow/llm"

block:
  putEnv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:9100/")
  putEnv("COWORLD_LLM_MODEL", "anthropic/claude-sonnet-4.6")
  putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://retired.invalid")
  putEnv("ANTHROPIC_API_KEY", "local-key-must-not-be-used")
  let client = newLlmClient(GameConfig())
  for slot in 0 .. 1:
    let request = client.requestFor("rules", "private view", slot)
    doAssert request.url == "http://127.0.0.1:9100/v1/messages"
    doAssert request.headers["X-Coworld-Player-Slot"] == $slot
    let body = parseJson(request.body)
    doAssert body["model"].getStr() == "anthropic/claude-sonnet-4.6"
    doAssert not body.hasKey("anthropic_version")
    doAssert not body.hasKey("output_config")
  echo "hosted sidecar routing and seat attribution passed"

block:
  delEnv("COWORLD_LLM_ENDPOINT")
  delEnv("COWORLD_LLM_MODEL")
  delEnv("BEDROCK_MODEL")
  putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://127.0.0.1:9200")
  putEnv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
  let client = newLlmClient(GameConfig())
  doAssert client.bedrockModelId == DefaultBedrockModel
  let expectedUrl = "http://127.0.0.1:9200/model/" & DefaultBedrockModel & "/invoke"
  doAssert client.requestFor("rules", "view", 0).url == expectedUrl

  # 429 throttle must fail cleanly without rotating to any alternate model.
  var raised429 = false
  try:
    discard client.textOf(Response(code: 429, body: "throttled"), "", expectedUrl)
  except LlmInfraError:
    raised429 = true
  doAssert raised429
  doAssert client.bedrockModelId == DefaultBedrockModel
  doAssert client.requestFor("rules", "view", 0).url == expectedUrl

  # 403 model access denied must disable client and not rotate to any alternate model.
  var raised403 = false
  try:
    discard client.textOf(Response(code: 403, body: "Model access is denied"), "", expectedUrl)
  except LlmInfraError:
    raised403 = true
  doAssert raised403
  doAssert client.disabled
  doAssert client.bedrockModelId == DefaultBedrockModel
  doAssert client.requestFor("rules", "view", 0).url == expectedUrl
  echo "bedrock single-model no-fallback behavior passed"
