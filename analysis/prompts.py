def generate_large_prompt(data_point):
    return f"""
            Below is a section of a national security document from the country {data_point["Country"]}. Focus on the country's position towards {data_point['TARGET']}. Based on this section of text, classify how the country {data_point['Country']} views {data_point['TARGET']} as either "Aligned", "Not_Aligned", or "Neutral/Irrelevant".

            Aligned: respond with "Aligned" if the text meets the following rules.
            -Indicates a desire for greater cooperation between the {data_point["Country"]} and {data_point['TARGET']}.
            -Includes a description of cooperative actions with {data_point['TARGET']}, even if the language itself is neutral, or if the activity took place in the past.
            -Includes praise or admiration for {data_point['TARGET']}.

            Not_Aligned: Respond with "Not_Aligned" if the text meets the following rules.
            -Includes a negative description of {data_point['TARGET']}.
            -Includes a description of non-cooperative, hostile action taken by {data_point['TARGET']}.
            -Mentions words such as competitor or threat in reference to {data_point['TARGET']}.

            Neutral/Irrelevant: Respond with "Neutral/Irrelevant" if the text meets the following rules.
            -Includes objective descriptions or factual statements about {data_point['TARGET']} without mentioning any cooperation or conflict.
            -Includes neutral language without any normative statements made about {data_point['TARGET']}.
            -Is too short or does not provide relevant information
            -Is a fragment or incomplete sentence yielding ambiguity about author intent.
            -It is a reference to another document or a part of an index, a table of contents, or a photo caption.
            -The text is not explicitly directed at {data_point['TARGET']}: for example, it only mentions {data_point['TARGET']} in passing without substantive information or statements.

            Respond with only your classification.
            Text: {data_point["TEXT"]}
            Classification: """.strip()

def generate_gpt_prompt(data_point):
    return f"""
            Classify the following text writen by {data_point["Country"]} about {data_point["TARGET"]} as one of three labels: ALIGNED, NOT_ALIGNED, or NEUTRAL/IRRELEVANT.

            Instructions:
            - 0 (NOT_ALIGNED): Text shows hostility, criticism, or conflict with {data_point["TARGET"]}.
            - 1 (ALIGNED): Text shows cooperation, praise, or joint actions with {data_point["TARGET"]}.
            - 2 (NEUTRAL/IRRELEVANT): Text is factual, ambiguous, or unrelated to {data_point["TARGET"]}.

            Focus ONLY on statements about {data_point["TARGET"]}. Respond with exactly one label.

            Example 1:
            Text: "{data_point["TARGET"]} engages in dialogue with {data_point["TARGET"]} on climate change issues."
            Classification: ALIGNED (1)

            Example 2:
            Text: "{data_point["TARGET"]} sees {data_point["TARGET"]}'s behavior as a challenge to its security."
            Classification: NOT_ALIGNED (0)

            Example 3:
            Text: "{data_point["TARGET"]} seeks to promote multipolarity in global security."
            Classification: NEUTRAL/IRRELEVANT (2)

            Text: {data_point["TEXT"]}
            Classification:""".strip()

def generate_simple_prompt(data_point):
    return f"""
            Below is a section of a national security document from the country {data_point["Country"]}. Focus on the country's position towards {data_point['TARGET']}. Based on this section of text, classify how the country {data_point['Country']} views {data_point['TARGET']} as either "Aligned", "Not_Aligned", or "Neutral/Irrelevant".
            Respond with only your classification.
            Text: {data_point["TEXT"]}
            Classification: """.strip()

def generate_encoder_prompt(data_point):
    """For standard encoder models (no NLI pretraining)."""
    return f"[Source: {data_point['Country']}] [Target: {data_point['TARGET']}] {data_point['TEXT']}"

def generate_nli_premise(data_point):
    """Premise for NLI text-pair input. Paired with TEXT as hypothesis."""
    return f"{data_point['Country']} views {data_point['TARGET']} as aligned with its interests."