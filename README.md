# VectorClues

VectorClues is a bot that uses OpenAI's embeddings model to create clues for the Codenames game.

### Setup
To set up you need to create the .env file and add your OpenAI API key.
```
OPENAI_API_KEY=<"Your-Key-Here">
```
Then you need to embed the provided dictionary. This will cost a few cents. To do so run:
```commandline
python3 embed_words.py run 
```
The embeddings are saved at ```data/default_embeddings.pkl ```.
You need to change the name of this file to ```data/58k_embeddings_meaning.pkl```.
Changing the name avoids overwriting the default embeddings.

### Run
To run the bot for a game, use:
```commandline
python3 play.py
```

To only run the bot once and with preset words, use:
```commandline
python3 sort_words.py run
```
### Limitations
Currently, the bot doesn't support multiple word phrases.
In the game this would usually be a proper noun.
There are also a few words in the game that aren't in the 58 thousand word dictionary.

The clues suffer mostly from the odd nature of embeddings' "closeness".
This can be best seen by examining the closest words to a given word.
You will notice that the closest words aren't what a human would choose.
To see the closest words to a given word, run the following (number of words to return as an optional parameter).
```
python3 sort_words.py get_closest <word> [number of words]
```
