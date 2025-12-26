
PROMPT_TEMPLATE = """
# Reflection Task:
You have just extracted team names and player names from the following sentence:


<extraction_data>
    Sentence: {sentence}
    Predicted Team Names: {predicted_team_names}
    Predicted Player Names: {predicted_player_names}
</extraction_data>

Did you make a mistake when extracting team and player names ? Consider:


# Potential errors to consider

## Review for False Negatives


Full team names (e.g., "Kings de Sacramento", "New Orleans/Oklahoma City Hornets").
Partial or colloquial team names (e.g., "New York" as a team, "Michigan" as a university team).
Teams mentioned in possessive or plural forms.


## Review for False Positives

Did you incorrectly label any non-player/non-team entities as players or teams? Consider:

- Question words (e.g., you retrieved "Qui", "Quel joueur").
- Locations, universities, or other non-entities (e.g., "Michigan", "New York hot dog vendor").
- Ambiguous terms that could be misinterpreted (e.g., "New York" as a city vs. a team).
- Inexact retrieval (e.g. the value in the original sentence is "Bulls de Chicago" but you extracted "Chicago Bulls")

## Contextual Check

Does the context of the sentence support the labels you assigned?

For example, is "New York" referring to a team or a location?
Is "Michigan" referring to a player or a university?

Are there any phrases that are clearly not players or teams but were labeled as such?


# Output

You must output a risk of error, as one of these categories:
- high: you are pretty sure there is an error
- medium: there is a risk of error but you're not sure
- low: the extraction is correct

The response must be given in json format, example: {{"risk":"high"}}
"""
