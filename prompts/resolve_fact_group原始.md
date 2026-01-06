You are a top AI narrative logician and memory consolidation expert. Your task is to analyze a list of unordered factual statements from a user's history that all relate to the same topic. Your goal is to construct the most coherent and current representation of the user's memory.

Follow these steps:
1.  **Analyze all facts**: Read all the provided statements.
2.  **Infer Chronological Order**: Based on common sense and logical progression, determine the most likely temporal sequence of these facts. This is your initial timeline.
3.  **Consolidate Memory & Resolve Conflicts (Crucial Update)**: Determine the most current and accurate state of the user's memory.
    - **Core Principle (KEEP Historical Context):** **Historical experiences ("experience"), background information** (e.g., "investment failure in 2020"), or **prior states** (e.g., "used to like tea") **must NOT be discarded**. These facts represent a chronological progression, not a logical conflict, when compared to current states (e.g., "currently has good investment returns"). They collectively form the complete memory evolution chain.
    - **Discard Condition (ONLY discard IF Mutually Exclusive):** Only discard an older fact if two or more facts are **logically Mutually Exclusive** and the **newest fact explicitly negates the older one**.
        - **Mutually Exclusive Example 1 (Dynamic Fact/Plan):** "Planned a trip to Japan" vs. "Canceled Japan trip, now planning Texas trip." The latest plan replaces the old one; the old plan is discarded.
        - **Mutually Exclusive Example 2 (Directly Opposing Preference):** "Likes coffee" vs. "Dislikes coffee" (assuming this represents an attitude change over time). **Always retain the newest preference** and discard the old, opposing one.
        - **Non-Mutually Exclusive Example (Experience vs. Current Status):** "Investment failure in 2020" vs. "Currently has good investment returns." **Both facts must be retained** and sorted chronologically.
    - **Result Handling:** All **retained** facts (including historical context) must appear in the final `sorted_source_ids` list. Only older facts that have been clearly replaced or negated should appear in `discarded_source_ids`.
4.  **Construct a Narrative**: In one or two sentences, summarize the final, consolidated state of this memory, emphasizing the evolution from past to present where applicable (e.g., from liking tea to preferring coffee now).
5.  **Output the Results**: Provide the final lists of kept and discarded `source_id`s.


**Output Format (Strictly JSON):**
Please provide your response ONLY in the following JSON format. Do not add any text before or after the JSON block.

```json
{{
    "narrative_summary": "A brief summary of the memory evolution and its final state, highlighting the progression (e.g., from liking tea to preferring coffee now).",
    "sorted_source_ids": ["id_of_kept_fact_1", "id_of_kept_fact_2", ...],
    "discarded_source_ids": ["id_of_discarded_fact_A", ...],
    "reasoning": "A short explanation of why you kept, sorted, and discarded the facts this way. Specifically mention which facts were retained because they are historical context and which were discarded because they are mutually exclusive."
    }}
```

**Input Facts:**
{data}
