You are tasked with constructing a **memory-based tool-use test scenario** for long-term memory evaluation.

### Input context

You are given:

- **conversation_history**: a complete JSON transcript of a past multi-turn conversation between a user and an AI assistant.  
  It may be long (e.g., ToolBench-like data) and could have occurred a long time ago.  
  The conversation includes natural dialogue and one or more assistant tool calls.
- **used_tools**: a JSON array describing all tools that the assistant has access to in this conversation.  
  Each tool has a "name" and an "argument schema" that defines which arguments it expects.

---

### Your goal

Generate a **single test case** that evaluates whether the assistant can recall a concrete value from the past conversation when forming a tool call.

The recalled value (e.g., an identifier, query string, file name, URL, or timestamp) must appear **verbatim somewhere in conversation_history**, but must **not** appear in the newly written user prompt.

The final output should simulate a realistic continuation of the conversation — as if the user is returning weeks later with a new request that implicitly depends on memory of prior tool use.

| Principle                                               | Description                                                                                                                       | Example                                                                                                 |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 🧩 **1. Continuation Instead of Recall**                | The user doesn’t say “Do you remember…”, but continues naturally using phrases like “that one,” “the same as before,” or “again.” | “Help me check the latest quarterly report for that company again.”                                     |
| 🕰️ **2. Time Gap but Task Continuity**                 | Use temporal anchors like “before,” “last time,” or “a while ago” to create a sense of time passing.                              | “Last time we looked at that company’s data — have there been any changes recently?”                    |
| 🧍 **3. Implicit Reference to Trigger Recall**          | Force the agent to search context to identify which entity the user refers to.                                                    | “Could you send me the newest financial report for that tech company we talked about earlier?”          |
| 🧠 **4. Task Continuation Instead of Knowledge Recall** | The request should be an action-oriented task, not a direct question about memory.                                                | “Analyze again the trend of that fund we discussed before.”                                             |
| 🎯 **5. Mix in Irrelevant Background Info**             | Insert small talk or unrelated context to make the interaction feel more realistic.                                               | “The market’s been chaotic lately. By the way, update the filings for that company we checked earlier.” |

---

### Strict output requirements

Return a single **JSON object** with exactly two keys: `"user_prompt"` and `"expected_tool_call"`.

1. **"user_prompt"**

   - Write a natural, realistic user message that implicitly refers to something from the conversation history (e.g., “that report we exported”, “the ID from before”, “the query you ran last time”).
   - It should *sound like a normal user continuation*, not like a test question.
   - It must **not contain or leak** the actual value being recalled — the assistant must retrieve it from memory.
   - The request should be plausible as a follow-up after a long delay (e.g., “a few weeks later”, “I’m continuing that project we started earlier”).

2. **"expected_tool_call"**

   - Must be a valid JSON object with two keys:

     ```json
     {
       "name": "<tool name>",
       "arguments": { "<argument>": "<recalled_value>" }
     }
     ```

   - `"name"` must exactly match one of the available tools listed in `used_tools`.

   - `"arguments"` must contain the **exact recalled value** as it appears in conversation_history, inserted in the correct field per the tool schema.

   - Do not fabricate new fields or extra arguments beyond what the tool schema supports.

3. **Output format**

   - Output **only** the final JSON object (no explanations, no code fences, no text before or after).

---
