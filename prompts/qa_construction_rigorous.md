# Memory-Driven Tool-Call QA Generation with Academic Rigor

You are an expert AI data generator specializing in **Tool Learning** and **Long-Term Memory** research.

Your task is to construct **academically rigorous, memory-dependent QA pairs** using the **Reverse-Masking Paradigm** with strict quality control.

---

## 🎯 Core Objective

Generate QA samples where:
1. **Necessity**: The question CANNOT be answered correctly without accessing the user's long-term memory
2. **Determinism**: Tool call arguments are VERIFIABLY anchored to specific facts in the memory evolution chain
3. **Naturalness**: The query is task-oriented and uses implicit references (pronouns, deixis) rather than explicit values

---

## 📋 Generation Pipeline (Reverse Masking)

### Step 1: Analyze Memory Evolution Chain

Carefully examine the provided memory chain to identify:
- **Explicit Facts**: Concrete information directly stated (e.g., "User prefers Japanese food")
- **Evolution Patterns**: How information changes over time (add/update/conflict)
- **Critical Parameters**: Values that ONLY exist in memory and would NOT be known from context

**Key Question**: What information is EXCLUSIVE to this memory and cannot be guessed from the query alone?

### Step 2: Construct Ground Truth Tool Call(s)

Design tool call(s) that satisfy:

1. **Memory Anchoring Constraint** (CRITICAL):
   - **At least ONE parameter** must have its value DIRECTLY taken from the memory chain
   - This parameter's value must be VERBATIM or a clear derivation from memory facts
   - Example: If memory says "User created folder 'sampleFolder'", tool arg must be `folder_name: "sampleFolder"`

2. **Task Realism**:
   - Other parameters should reflect realistic user intent
   - Support multi-step tasks if needed (e.g., [check_status, turn_off])

3. **Schema Compliance**:
   - Ensure tool name and argument structure are valid
   - Use proper data types (string, int, boolean, array)

### Step 3: Generate Masked Query (Task-Oriented)

Create a natural user query that:

1. **Implicitly References Memory** (REQUIRED):
   - Use pronouns: "my folder", "that file", "the one I mentioned"
   - Use deixis: "the room I just used", "the person from yesterday"
   - Use vague descriptors: "my usual place", "the recent project"

2. **Does NOT Leak the Answer**:
   - ❌ BAD: "Turn off the light in the multi-purpose room" (explicitly states "multi-purpose room")
   - ✅ GOOD: "Turn off the light in the room I just used" (requires memory to resolve "the room")

3. **Is Task-Oriented**:
   - Describes a goal or intent
   - Sounds like a real user request
   - May involve multiple sub-tasks

### Step 4: Self-Verification Checklist

Before outputting, verify:

- [ ] **Anchoring**: At least one tool argument value appears VERBATIM in memory facts
- [ ] **Masking**: The query does NOT contain the memory-anchored value(s)
- [ ] **Necessity**: Without memory, an agent would have to GUESS the anchored parameter
- [ ] **Executability**: The tool call is complete and properly formatted

---

## 🎚️ Difficulty Level Definitions

Your generated sample will be automatically classified, but understanding these helps:

### Level 1: Explicit Retrieval
- Memory chain has no conflicts or updates
- Single clear fact: "User created folder X"
- Query directly refers to this: "Delete the folder I just created"
- **Anchored parameter**: Exactly matches the fact

### Level 2: Implicit/Evolution
- Memory evolves: "User initially preferred Chinese food, but now prefers Japanese food"
- Query requires distinguishing current vs. obsolete: "Book a table at my favorite type of restaurant"
- **Anchored parameter**: Must use LATEST state
- May involve pronouns or temporal references

### Level 3: Conflict/Multi-Hop
- Memory has explicit contradictions: "User said no spicy food, BUT later said loves spicy food"
- Query involves conflict resolution: "Order food following my dietary preferences"
- **Anchored parameter**: Requires reasoning across multiple facts
- May require aggregating information from multiple memory slots

---

## 📤 Output Format

Return a JSON object with the following structure:

```json
{{
  "tool_name": "<primary_tool_name>",
  "masked_query": "<task-oriented natural language query with implicit memory references>",
  "ground_truth_call": [
    {{
      "tool": "<tool_name>",
      "arguments": {{
        "<param_1>": "<value_from_memory>",
        "<param_2>": "<value_from_query_or_inference>",
        ...
      }}
    }},
    ...
  ],
  "implicit_memory_arguments": {{
    "<param_name>": "<exact_value_from_memory>",
    ...
  }},
  "explicit_arguments": {{
    "<param_name>": "<value_inferrable_from_query>",
    ...
  }},
  "difficulty_level": <1|2|3>,
  "memory_chain_analysis": "<Explanation of how memory facts are used to resolve implicit references>"
}}
```

### Field Specifications

- **`tool_name`**: The primary tool being called (string)
- **`masked_query`**: Natural language query (string)
  - Must use implicit references (pronouns, deixis)
  - Must NOT contain values from `implicit_memory_arguments`
  
- **`ground_truth_call`**: Array of tool call objects (even if single call)
  - Each object has: `{{"tool": "...", "arguments": {{...}}}}`
  
- **`implicit_memory_arguments`**: Dict of parameters that REQUIRE memory
  - Key: parameter name
  - Value: EXACT value from memory facts (verbatim or clear derivation)
  
- **`explicit_arguments`**: Dict of parameters inferrable from query
  - Key: parameter name
  - Value: Value that can be determined without memory
  
- **`difficulty_level`**: Integer 1-3 (your suggestion, will be verified programmatically)

- **`memory_chain_analysis`**: String explaining the reasoning
  - Which memory facts are used?
  - How does the query's implicit reference map to memory?
  - What would go wrong without memory?

---

## ⚠️ Critical Constraints (Will Be Auto-Validated)

1. **Argument Grounding Validation**:
   - Your `implicit_memory_arguments` values will be checked against memory facts
   - If they don't appear in the memory chain, your sample will be REJECTED
   
2. **Adversarial Filtering** (The "Necessity Test"):
   - A memory-less LLM will attempt to answer your query
   - If it can guess the correct arguments, your sample will be REJECTED
   - This forces you to properly mask memory-dependent information
   
3. **No Hallucination**:
   - Do NOT invent facts not present in the memory
   - Do NOT generate tool calls with arbitrary values
   - Every critical parameter must be traceable to memory

---

## 📝 Example: Multi-Step Task (Smart Home Control)

### Input Memory Chain:
```json
{{
  "evolution_chain": [
    {{
      "fact": "User stopped the washing machine in the multi-purpose room",
      "evolution_type": "add",
      "long_term_memory_point": "User can control the washing machine in the multi-purpose room"
    }},
    {{
      "fact": "User checked if the washing machine in the multi-purpose room is powered off",
      "evolution_type": "add",
      "long_term_memory_point": "User can control the washing machine in the multi-purpose room and check its power status"
    }},
    {{
      "fact": "User turned off the light in the multi-purpose room",
      "evolution_type": "add",
      "long_term_memory_point": "User can control the washing machine and light in the multi-purpose room"
    }}
  ],
  "current_fact": "User can control the washing machine and light in the multi-purpose room",
  "long_term_memory_point": "User can control the washing machine and light in the multi-purpose room"
}}
```

### Expected Output:
```json
{{
  "tool_name": "control_device",
  "masked_query": "I'm about to leave. Help me completely shut down the room where I just did laundry - make sure that washing machine is fully stopped and powered off, and turn off the light there too.",
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
  "memory_chain_analysis": "The query uses 'the room where I just did laundry' to implicitly reference the 'multi-purpose room' mentioned repeatedly in the memory chain (S1-S3). Without memory, an agent would not know which specific room the user is referring to. The devices (washing machine, light) and actions are explicitly stated in the query, so they don't require memory."
}}
```

**Why This Is Valid**:
- ✅ `room: "multi-purpose room"` is anchored to memory (appears in all 3 facts)
- ✅ Query says "the room where I just did laundry" (implicit) instead of "multi-purpose room" (explicit)
- ✅ Without memory, agent cannot determine which room to control
- ✅ Multi-step task is realistic and properly structured

---

## 🚫 Common Mistakes to Avoid

1. **Leaking Memory in Query**:
   - ❌ "Turn off the light in the multi-purpose room"
   - ✅ "Turn off the light in the room I just used"

2. **Using Arbitrary Values**:
   - ❌ `{{"location": "downtown"}}` when memory doesn't mention "downtown"
   - ✅ Only use values that are grounded in memory facts

3. **Trivial Masking**:
   - ❌ Query: "Do the thing I mentioned" (too vague, agent can't infer task)
   - ✅ Query: "Book a table at my usual restaurant" (clear task, implicit parameter)

4. **No Memory Dependency**:
   - ❌ Query: "What's the weather in New York?" (no memory needed)
   - ✅ Query: "What's the weather in the city I'm visiting?" (memory: "User is visiting New York")

---

## 📥 Your Task

Given the following memory evolution chain, generate a high-quality, memory-dependent QA pair following all constraints above:

### Input Data:

**Memory Evolution Chain**:
{long_term_memory_point}


**Instructions**:
1. Identify which facts contain information that MUST come from memory
2. Design a tool call with at least one memory-anchored parameter
3. Generate a natural query that masks this parameter using implicit references
4. Provide complete JSON output following the schema above

Generate your response now:
