// Phase 14 - Lesson 18, exercise 2: the lesson 01 ReAct loop on Mastra.
//
// Needs Node, @mastra/core, zod and a model key. It follows the
// createTool and Agent examples on Mastra's "using tools" page.
//
// Compare with lesson 01's main.py: the loop, the registry and the dispatch
// are gone, and each tool now carries a Zod schema. That schema validates the
// model's arguments at runtime, is what the model is shown as the tool's
// input schema, and types the argument of execute() at compile time.

import { Agent } from '@mastra/core/agent'
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'

const store = new Map<string, string>()

export const calculator = createTool({
  id: 'calculator',
  description: "Evaluate an arithmetic expression such as '120 * 0.15'",
  // lesson 01 checks the characters by hand inside the tool; here the schema does it
  inputSchema: z.object({ expr: z.string().regex(/^[0-9+\-*/(). ]+$/) }),
  outputSchema: z.object({ result: z.string() }),
  execute: async ({ expr }) => ({ result: String(Function(`"use strict"; return (${expr})`)()) }),
})

export const kvSet = createTool({
  id: 'kv_set',
  description: 'Store a value under a key for later steps',
  inputSchema: z.object({ key: z.string(), value: z.string() }),
  outputSchema: z.object({ result: z.string() }),
  execute: async ({ key, value }) => {
    store.set(key, value)
    return { result: `stored ${key}` }
  },
})

export const kvGet = createTool({
  id: 'kv_get',
  description: 'Read back a value stored with kv_set',
  inputSchema: z.object({ key: z.string() }),
  outputSchema: z.object({ result: z.string() }),
  execute: async ({ key }) => ({ result: store.get(key) ?? `missing:${key}` }),
})

export const taxAgent = new Agent({
  id: 'tax-agent',
  name: 'Tax Agent',
  instructions:
    'Use the calculator for all arithmetic. Store intermediate values with kv_set before the final answer.',
  model: process.env.MASTRA_MODEL ?? '', // a "provider/model" string from Mastra's model router
  tools: { calculator, kvSet, kvGet },
})

// const result = await taxAgent.generate('What is 120 plus 15% tax, stored in kv?')
// console.log(result.text)
