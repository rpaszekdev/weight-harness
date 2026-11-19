"""hello-llm v2 lexicon: the words the model knows that are not facts.

Word classes, cue words and the 7 relation frames (how a fact is said in English).
No answers live here. Facts come from facts.json; build.py compiles both into weights.
"""
TYPES = ["country", "city", "continent", "language", "currency", "animal", "sound", "class", "number", "habitat", "diet", "adj"]
SUBJ_TYPES = ["country", "animal", "adj"]
OBJ_TYPES = ["city", "continent", "language", "currency", "sound", "class", "number", "habitat", "diet", "adj"]
LIST_TYPES = {  # types you can ask for a whole list of, and the plural nouns that ask for it
    "country": "countries nations", "city": "capitals cities", "animal": "animals creatures",
    "language": "languages", "continent": "continents", "currency": "currencies",
}
LIST_SINGULAR = {"country": "country nation", "city": "capital city", "animal": "animal creature",  # "every animal"
                 "language": "language", "continent": "continent", "currency": "currency"}

RELATIONS = {  # relation: (subject type, object type, can be read backwards)
    "capital": ("country", "city", True),
    "continent": ("country", "continent", False),
    "language": ("country", "language", False),
    "currency": ("country", "currency", False),
    "sound": ("animal", "sound", True),
    "legs": ("animal", "number", False),
    "class": ("animal", "class", False),
    "habitat": ("animal", "habitat", False),
    "diet": ("animal", "diet", False),
    "opposite": ("adj", "adj", True),
}
PRIORITY = ["none", "continent", "habitat", "sound", "legs", "language", "currency", "diet", "class", "capital",
            "opposite"]  # tie order

FRAMES = {  # S/O = subject/object word, AS/AO = "a" or "an" for it
    "capital": "the capital of S is O .",
    "continent": "S is in O .",
    "language": "people in S speak O .",
    "currency": "the currency of S is the O .",
    "sound": "AS S says O .",
    "legs": "AS S has O legs .",
    "class": "AS S is AO O .",
    "habitat": "the home of AS S is the O .",
    "diet": "AS S is AO O .",
    "opposite": "the opposite of S is O .",
}
GREET_PREFIX = "hello world !"
YES_PREFIX, NO_PREFIX = "yes ,", "no ,"
UNKNOWN_LEAD = "i don't know about"  # + the unknown word, copied from the question
IDK = "i don't know ."

CUES = {  # cue channel: (relation it votes for, votes, words)
    "capital": ("capital", 2, "capital capitals capitol government"),
    "continent": ("continent", 2, "continent continents located"),
    "where": ("continent", 1, "where wheres"),
    "language": ("language", 2, "language languages speak speaks spoken speaking tongue"),
    "sound": ("sound", 2, "sound sounds noise noises say says said make makes"),
    "legs": ("legs", 2, "legs leg limbs"),
    "currency": ("currency", 2, "currency currencies money cash pay use uses used"),
    "habitat": ("habitat", 2, "live lives living habitat home found"),
    "diet": ("diet", 2, "eat eats eating eater eaters diet food feed feeds"),
    "class": ("class", 2, "class classes category"),
    "kind": ("class", 1, "kind kinds type types sort group"),
    "opposite": ("opposite", 2, "opposite opposites antonym antonyms contrary"),
}
TYPE_VOTES = {  # a mentioned entity of this type votes for a relation
    "country": ("capital", 1), "city": ("capital", 2), "continent": ("continent", 2),
    "language": ("language", 2), "currency": ("currency", 2), "animal": ("class", 1), "sound": ("sound", 2),
    "class": ("class", 2), "number": ("legs", 2), "habitat": ("habitat", 2), "diet": ("diet", 2), "adj": ("opposite", 1),
}

GREET_WORDS = "hello hi hey hiya heya howdy greetings yo hola sup"
GREET_PHRASES = ["good morning", "good afternoon", "good evening"]
YN_WORDS = "is are am do does did was were can has have would could should will"
FUNCTION_WORDS = """
a an the and or but if so of in on at to from by for with about as into onto over under than
this that these those there here it its i me my mine you your yours he him his she her we us our they them their
what which who whom whose when where why how whats hows whos many much more most some any all each every both either
no not yes yeah yep nope ok okay be been being had having could will would should shall may might must
dont doesnt didnt isnt arent wasnt cant wont s t re ll ve d m
tell please know think name named called call give list show find get go goes going make makes made like
also just only very really animal animals creature creatures country countries city cities nation nations state
place places world thing things people person someone everyone one ones number numbers letter letters word words
exactly usually typically normally often always ever generally actually commonly mainly mostly primarily
main official national common known typical hear heard u ur r pls plz
above across after again against ago almost alone along already although among another answer anybody anyone anything
anyway around ask asked asking away back basically because become before behind believe below best better between
beyond bit bring buy came care case certain certainly change check choose clear come comes coming consider count data
dataset database describe detail details different down during earlier easily else enough entire especially even
everything example except explain fact facts fairly feel few first following friend full further gets getting given
gives gimme glad gone got great guess guys happen happened hope however idea im important include included includes
including indeed info information inside instead interested itself keep knew knows last later least less let lets
likely listed little look looking lot lots maybe mean means mention mentioned mind myself need needed needs neither
never next nice nobody none nor nothing now off oh once onto other others otherwise out own part particular perhaps
plus possible possibly probably provide put quick quickly quite rather reason recall remember represent represented
right same see seem seems seen set several share since something sometimes somewhat soon sorry still stuff such
suppose take tells thank thanks thats then therefore though thought through together too toward true try trying
unless until upon usually various want wanted wants way ways well whatever whether while whole wish within without
wonder wondering belong belongs using give list name"""

SYNONYMS = {  # (type, entity): other words for it
    ("country", "united states"): ["usa", "america"],
    ("country", "united kingdom"): ["uk", "britain", "great britain"],
    ("country", "south korea"): ["korea"],
    ("city", "new delhi"): ["delhi"],
    ("continent", "asia"): ["asian"], ("continent", "europe"): ["european"], ("continent", "africa"): ["african"],
    ("continent", "north america"): ["north american"], ("continent", "south america"): ["south american"],
    ("continent", "oceania"): ["oceanian"],
    ("diet", "herbivore"): ["herbivorous"], ("diet", "carnivore"): ["carnivorous"], ("diet", "omnivore"): ["omnivorous"],
    ("diet", "insectivore"): ["insectivorous"],
    ("language", "chinese"): ["mandarin"],
    ("number", "0"): ["zero", "no legs"],
    ("animal", "hippopotamus"): ["hippo"], ("animal", "rhinoceros"): ["rhino"],
    ("number", "2"): ["two"], ("number", "4"): ["four"], ("number", "6"): ["six"], ("number", "8"): ["eight"],
}
IRREGULAR_PLURALS = {"mouse": "mice", "wolf": "wolves", "goose": "geese", "sheep": "sheep", "goldfish": "goldfish",
                     "fish": "fishes"}
PLURAL_TYPES = ["animal", "class", "habitat", "diet"]
MAX_PHRASE = 3  # longest multi-word token, e.g. "dominican republic"

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
PUNCT = ".,!?'-:;()"


def plural(word):
    return IRREGULAR_PLURALS.get(word, word + ("es" if word.endswith(("s", "sh", "ch", "x", "z")) else "s"))


def verb_forms(word):
    """bark -> barks, barking, barked (sounds are verbs too: 'which animal barks?')."""
    s = word + ("es" if word.endswith(("s", "sh", "ch", "x", "z")) else "s")
    return [s, word + "ing", word + "ed"]
