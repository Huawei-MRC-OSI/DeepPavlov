from deeppavlov.dataset_readers.snips_reader import SnipsReader
from deeppavlov.dataset_iterators.snips_intent_slot_iterator import SnipsIntentSlotIterator
from deeppavlov.models.preprocessors.str_lower import StrLower
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.models.snips.intent_slot_classifier import IntentSlotClassifier

data_reader = SnipsReader().read("/workspace/snips-dataset")
train_iterator = SnipsIntentSlotIterator(data=data_reader)

tokens_vocab = SimpleVocabulary(
    save_path='/workspace/snips-dataset/tokens.dict',
    load_path='/workspace/snips-dataset/tokens.dict',
    min_freq=2,
    special_tokens=('<PAD>', '<UNK>'),
    unk_token='<UNK>')
intents_vocab = SimpleVocabulary(
    save_path='/workspace/snips-dataset/intents.dict',
    load_path='/workspace/snips-dataset/intents.dict')
slots_vocab = SimpleVocabulary(
    save_path='/workspace/snips-dataset/slots.dict',
    load_path='/workspace/snips-dataset/slots.dict')

train_data = train_iterator.get_instances(data_type='train')
train_texts = train_data[0]
train_intents, train_slots = list(zip(*train_data[1]))
str_lower = StrLower()

tokens_vocab.fit((str_lower(train_texts)))
intents_vocab.fit((train_intents))
slots_vocab.fit((train_slots))

tokens_vocab.save()
intents_vocab.save()
slots_vocab.save()

model = IntentSlotClassifier()

for epoch in range(10):
    for batch in train_iterator.gen_batches(batch_size=32, data_type="train"):
        x = batch[0]
        y_intents, y_slots = list(zip(*batch[1]))

        x = tokens_vocab(str_lower(x))
        y_intents = intents_vocab(y_intents)
        y_slots = slots_vocab(y_slots)

        model.train_on_batch(x, y_intents, y_slots)

