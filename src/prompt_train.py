TRAIN_ASSISTANT_PROMPT_NER_RETRIEVAL = """
# Instructions
You are given a text that may contain some NBA players and players and teams.
Retrieve the list of players and teams from the text.

# Constraints
- DO NOT MODIFY THE NAMES OF THE PLAYERS AND TEAMS.
- USe the same case as in the input text.
- If there is no player or team in the text, return an empty list for that category
- Some names may contain typos. Still, try to retrieve them as-is from the text.

# Examples

Example n°1 :
Input: "How many rebounds did mike pietrus have in the 2018 playoffs?"
Output: {{'players': ['mike pietrus'], 'teams': []}}

Example n°2 :
Input: "How many points did Victor wembanyama have in the 2024 season for the spurs?"
Output: {{'players': ['Victor wembanyama'], 'teams': ['spurs']}}

Example n°2 :
Input: "The best player of the 2011-2012 season is Lebrone james."
Output: {{'players': ['Lebrone james'], 'teams': []}}

Example n°3 :
Input: "Y'all say the Minnesota Timberwolves are a rising force? Bro, even the Hornets from New Orleans/Oklahoma City had better chemistry than your team's last season."
Output: {{'players': [], 'teams': ["Minnesota Timberwolves", "Hornets", "New Orleans", "Oklahoma City"]}}


Notice that the name of the player and team is not modified, even if there is a typo.
"""  # noqa: E501
