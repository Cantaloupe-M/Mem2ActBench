You are an expert AI Data Generator specializing in 'Tool Learning' and 'Long-term Memory' scenarios.
Your task is to construct a high-quality task-oriented QA dataset using the **Reverse-Masking Paradigm**.

### Core Paradigm: Task-Oriented + Memory-Driven
Unlike the traditional "Question -> Answer" pattern, you need to work backwards:

1. **Analyze Memory Context**:
- Carefully read the 'Memory Evolution Chain'
- Identify which information exists only in historical memory and will not be explicitly mentioned in the current question

2. **Construct Task-Oriented Tool Calls**:
- Create a complete tool call `C_gt` (Ground Truth Call), where:
    * **At least one parameter $p_{{mem}}$** must strictly come from **memory**, exactly matching the provided `implicit_memory_arguments` values
    * **Other parameters $A_{{curr}}$** are inferred or created from current task intent, forming realistic user requirement scenarios
- If the task requires multiple steps, generate multiple tool calls (e.g., "turn off washing machine + turn off light" in the example)

3. **Generate Task-Oriented User Question $Q$**:
- The question must be **natural and task-driven**, describing the goal the user wants to accomplish
- **Key Constraint**: The question must use **pronouns or vague descriptions** to replace $p_{{mem}}$, for example:
    * "the room I just used" instead of "multi-purpose room"
    * "my usual store" instead of "Starbucks on 5th Ave"
    * "the person I mentioned last time" instead of "John Smith"
- The question should contain sufficient contextual clues for the Agent to recover specific values from memory
- **Strictly Forbidden**: The question must NOT contain the specific value of $p_{{mem}}$

4. **JSON Output Specification**:
- `implicit_memory_arguments`: Must copy the provided values verbatim, no modifications
- `explicit_arguments`: Only include parameters that can be directly inferred from the current question
- `ground_truth_call`: Complete tool call(s), can be a single object or an array
- If the task requires multiple steps, generate multiple tool calls
- Provide `memory_chain_analysis` explaining how to recover parameters from memory

### Example Reference (Washing Machine + Light Control Scenario)

**Memory Slots**:
- S1: Stop the washing machine in the multi-purpose room
- S2: Check if the washing machine in the multi-purpose room is powered off
- S3: Turn off the light in the multi-purpose room
- S4: User can control the washing machine and light in the multi-purpose room

**Task-Oriented Question**:
"I'm about to leave, help me completely shut down the room where I just did laundry:
Make sure that washing machine is fully stopped and powered off, and turn off the light there too."

**Tool Calls**:
```json
[
{{
    "tool": "control_device",
    "args": {{
    "room": "multi-purpose room",      // ← Memory-driven parameter (recovered from S1-S4)
    "device": "washing_machine",
    "action": ["stop", "check_power_off"]
    }}
}},
{{
    "tool": "control_device",
    "args": {{
    "room": "multi-purpose room",      // ← Also depends on memory
    "device": "light",
    "action": "turn_off"
    }}
}}
]
```

**Key Points**:
- The question only says "the room where I just did laundry" and "the light there", without explicitly stating "multi-purpose room"
- The `room` parameter must be recovered from memory slots S1-S4
- `device` and `action` can be directly inferred from the current task semantics

### Difficulty Level Definitions:
- **Level 1 (Explicit Retrieval)**: Memory facts are static and clear, no conflicts in history, direct query suffices
- **Level 2 (Evolution/Conflict)**: Memory changes over time (e.g., "moved from Boston to Seattle"), the question requires the latest state, must distinguish valid from obsolete facts
- **Level 3 (Cross-Domain/Reasoning)**: Memory parameters require aggregating multiple facts or inferring from previous tool outputs


### Output Requirements:
Please generate a JSON response ensuring:
1. `masked_query` is a task-oriented natural question, using pronouns to mask values in `implicit_memory_arguments`
2. `explicit_arguments` contains parameters that can be inferred from the current question
3. `ground_truth_call` is a complete executable tool call (can be a single object or an array)
4. If the task requires multiple steps, generate multiple tool calls
5. Provide `memory_chain_analysis` explaining how to recover parameters from memory

### JSON Output Format Example:
```json
{{
"tool_name": "control_device",
"masked_query": "I'm about to leave, help me completely shut down the room where I just did laundry: Make sure that washing machine is fully stopped and powered off, and turn off the light there too.",
"ground_truth_call": [
    {{
    "tool": "control_device",
    "arguments": {{
        "room": "multi-purpose room",
        "device": "washing_machine",
        "action": ["stop", "check_power_off"]
    }}
    }},
    {{
    "tool": "control_device",
    "arguments": {{
        "room": "multi-purpose room",
        "device": "light",
        "action": "turn_off"
    }}
    }}
],
"implicit_memory_arguments": {{
    "room": "multi-purpose room"
}},
"explicit_arguments": {{
    "device": ["washing_machine", "light"],
    "action": ["stop", "check_power_off", "turn_off"]
}},
"difficulty_level": 1,
"memory_chain_analysis": "The user mentioned 'multi-purpose room' multiple times in memory slots S1-S4 for washing machine and light control. The current question uses 'the room where I just did laundry' and 'the light there' as references, requiring recovery of the specific room name 'multi-purpose room' from memory."
}}
```

### Input Data

**Memory Evolution Chain (Historical Changes)**:

{long_term_memory_point}

