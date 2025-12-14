# ruff: noqa: E501
prompts_synthetic_data_generation = {
    "General NBA Mentions": """Generate 15 synthetic sentences that mention NBA players and/or teams. Include a mix of correct and rarely slightly misspelled names (e.g., 'LeBron James' as 'Lebron Jammes' or 'Lakers' as 'Lakrs'). The sentences should be casual, as if written by fans on social media or in forums. Cover a variety of contexts like game performances, trades, injuries, and opinions.""",
    "Game Performance Context": """Create 15 sentences about NBA players performing in games. Some names can rarely be misspelled (e.g., 'Stephen Curry' as 'Stefen Currey' or 'Bucks' as 'Bucs'). Include both positive and negative comments, such as praise for a player's performance or criticism of a team's loss.""",
    "Trade Rumors and Speculation": """Write 15 sentences about NBA trade rumors. Use a mix of correct and incorrect spellings for player and team names (e.g., 'Jayson Tatum' as 'Jayason Tatum' or 'Boston Celtics' as 'Boston Celtix'). The sentences should reflect speculation, excitement, or disappointment about potential trades.""",
    "Injury Updates": """Generate 15 sentences discussing NBA player injuries. Include minor spelling errors in player and team names (e.g., 'Kevin Durant' as 'Kevn Durant' or 'Nets' as 'Netz'). The sentences should convey concern, updates, or reactions to injury news.""",
    "Fan Opinions and Hot Takes": """Produce 15 social media-style sentences where fans express strong opinions about NBA players or teams. Use sometimes some misspellings (e.g., 'Nikola Jokić' as 'Nikola Jokic' or 'Nuggets' as 'Nugits'). The tone should be passionate, humorous, or controversial.""",
    "Historical References": """Create 15 sentences referencing historical NBA moments or great players. You can include minor spelling mistakes (e.g., 'Michael Jordan' as 'Micheal Jorden' or 'Bulls' as 'Buls'). The sentences should sound like nostalgic or educational comments.""",
    "Fantasy Basketball Context": """Write 15 sentences about fantasy basketball, mentioning NBA players and teams. Use some misspellings (e.g., 'Giannis Antetokounmpo' as 'Gianis Antetokounpo' or 'Clippers' as 'Clipers'). The sentences should discuss draft picks, player stats, and lineup decisions.""",
    "Rivalry and Trash Talk": """Generate 15 sentences reflecting NBA rivalries or trash talk between fans. Include misspellings (e.g., 'Kawhi Leonard' as 'Kawhi Lenard' or 'Raptors' as 'Rapters'). The tone should be competitive, playful, or sarcastic.""",
    "NBA Statistics with NBA and Teams": """Generate 15 questions about NBA statistics that explicitly mention both the NBA and specific team names. Include a mix of correct and occasionally misspelled team names (e.g., 'Warriors' as 'Warriars' or '76ers' as '76rs'). The questions should ask about player stats, team records, or comparisons between teams, as if asked by fans or analysts.""",
    "NBA Statistics with NBA or Teams": """Create 15 questions about NBA statistics where each question mentions either the NBA or specific team names, but not both. Use minor spelling errors for team names (e.g., 'Heat' as 'Heet' or 'Mavericks' as 'Mavricks'). The questions should focus on individual player achievements, team performances, or league-wide trends.""",
    "NBA Statistics without NBA or Teams": """Write 15 questions about NBA statistics without directly mentioning the NBA or any team names. Refer to players, positions, or general basketball terms (e.g., 'Who holds the record for most assists in a single season?' or 'Which player averaged a triple-double last year?').""",
}

prompt_entities_example = """ Here is a list of player and team names that you can use. The list is non exhaustive an while you should prefer refer to these players/teams, it's ok if you refer to players/teams not present in the list:

player_names: {player_names_list}.
team_names: {team_names_list}.
"""

prompt_general_instructions = "Don't add emojis or hashtags"
prompt_language = "**Important**: Generate the sentences in {language}."
