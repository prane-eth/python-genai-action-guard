import pytest
from google import genai
from google.genai import types

from google.genai._extra_utils import get_function_response_parts, get_function_response_parts_async
from google.genai.types import Candidate
from google.genai.types import Content
from google.genai.types import FunctionCall
from google.genai.types import FunctionResponse
from google.genai.types import GenerateContentResponse
from google.genai.types import Part


def test_interaction_create_with_action_guard(client):
    def my_guard(tool_call: types.FunctionCall) -> types.GuardDecision:
        return types.GuardDecision.ALLOW

    try:
        client.interactions.create(
            model='gemini-2.5-flash',
            input='Test',
            action_guard=my_guard,
        )
    except Exception as e:
        pass

@pytest.mark.asyncio
async def test_async_interaction_create_with_action_guard(client):
    try:
        await client.aio.interactions.create(
            model='gemini-2.5-flash',
            input='Test',
            action_guard=lambda _: types.GuardDecision.ALLOW,
        )
    except Exception as e:
        pass


def test_action_guard_blocks_sync():
  def func_under_test(a: int) -> int:
    return a + 1

  # Simulate a GenerateContentResponse returned by the model containing
  # a tool execution request (FunctionCall) for `func_under_test`
  response = GenerateContentResponse(
      candidates=[
          Candidate(
              content=Content(
                  parts=[
                      Part(
                          function_call=FunctionCall(
                              name='func_under_test',
                              args={'a': 1},
                          )
                      )
                  ]
              )
          )
      ]
  )
  function_map = {'func_under_test': func_under_test}
  
  # When the action_guard returns BLOCK, we expect the resulting
  # FunctionResponse to bypass actual execution and return this predefined error
  expected_parts = [
      Part(
          function_response=FunctionResponse(
              name='func_under_test',
              response={'error': 'Action blocked by action_guard'},
          )
      )
  ]

  def guard(tool_call: types.FunctionCall) -> types.GuardDecision:
    return types.GuardDecision.BLOCK

  actual_parts = get_function_response_parts(response, function_map, action_guard=guard)

  # Verify the actual returned object matches the expected error FunctionResponse
  for actual_part, expected_part in zip(actual_parts, expected_parts):
    assert actual_part.model_dump_json(
        exclude_none=True
    ) == expected_part.model_dump_json(exclude_none=True)


@pytest.mark.asyncio
async def test_action_guard_blocks_async():
  async def func_under_test(a: int) -> int:
    return a + 1

  async def guard(tool_call: types.FunctionCall) -> types.GuardDecision:
    return types.GuardDecision.BLOCK

  # Simulate a model generating an async FunctionCall for `func_under_test`
  response = GenerateContentResponse(
      candidates=[
          Candidate(
              content=Content(
                  parts=[
                      Part(
                          function_call=FunctionCall(
                              name='func_under_test',
                              args={'a': 1},
                          )
                      )
                  ]
              )
          )
      ]
  )
  function_map = {'func_under_test': func_under_test}
  
  # When the async action_guard returns BLOCK, we expect the resulting
  # FunctionResponse to bypass actual execution and return this predefined error
  expected_parts = [
      Part(
          function_response=FunctionResponse(
              name='func_under_test',
              response={'error': 'Action blocked by action_guard'},
          )
      )
  ]
  actual_parts = await get_function_response_parts_async(
      response, function_map, action_guard=guard
  )

  # Verify the actual returned object matches the expected error FunctionResponse
  for actual_part, expected_part in zip(actual_parts, expected_parts):
    assert actual_part.model_dump_json(
        exclude_none=True
    ) == expected_part.model_dump_json(exclude_none=True)
