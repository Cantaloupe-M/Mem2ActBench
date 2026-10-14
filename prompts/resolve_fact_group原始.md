You organize a group of user facts into a locally consistent memory evolution chain.
Each input item has a source_id and facts from one original conversation. The items
share a semantic topic, but may describe different entities, scopes, or times.

Apply these rules:
1. Order retained source IDs from earlier to later using explicit temporal cues,
   corrections, and logically supported state changes. Never treat source ID numbers
   or dataset names as timestamps. When no ordering evidence exists, preserve input
   order; do not invent dates, events, or transitions.
2. Retain compatible information, including specificity refinements (sports to
   basketball, Asian food to sushi) and valid changes over time (successive residences).
   An earlier state may remain as historical context when it adds meaningful evolution.
3. An explicit correction of the same preference or plan supersedes the old value
   ("aisle seat" followed by "actually, change it to window"). Discard the obsolete
   source only if it supplies no other useful compatible fact or historical context.
4. Distinguish entity and task scopes. A hotel budget of $200 and a flight budget of
   $500 are separate facts, not conflicting values to merge or overwrite. Retain both.
5. Discard off-context statements, strong logical contradictions that cannot be
   explained by a supported update or scope distinction, and near-duplicates with no
   information gain. Use explicit corrections or temporal evidence to choose between
   conflicting values; do not arbitrarily assume one is newer or more trustworthy.
   If the evidence cannot resolve a conflict, state the uncertainty in the summary
   rather than inventing a resolution.
6. Decide at source_id granularity: never split, rewrite, or invent source IDs. Consider
   all facts of a source before discarding it. Every input source_id must appear exactly
   once in either sorted_source_ids or discarded_source_ids, never in both.
7. Write a concise narrative_summary of the supported evolution and current state.
   In reasoning, briefly justify ordering and each discard, citing source IDs and
   relevant evidence. Describe independent scopes separately and mark uncertainty.

Return only one valid JSON object with exactly these four fields, without Markdown:
{{
  "sorted_source_ids": ["source_earlier", "source_later"],
  "discarded_source_ids": [],
  "narrative_summary": "Concise evidence-supported summary",
  "reasoning": "Brief justification of ordering, retention, and discards"
}}

Input facts:
{data}
