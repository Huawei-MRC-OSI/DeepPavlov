# Copyright 2019 Sergey Mironov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nltk
from typing import Tuple, List

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

WordRepr = str
Intent = str
SlotRepr = str
SentenceRepr = Tuple[List[WordRepr],Tuple[Intent,List[SlotRepr]]]

@register('snips_intent_slot_iterator')
class SnipsIntentSlotIterator(DataLearningIterator):
    @overrides
    def preprocess(self, data, *args, **kwargs)->List[SentenceRepr]:
        """
        Return list of 'sentences', where each 'sentence' is described by a
        list of words and a list of BIO-encoded slots.

        TODO: Override `gen_batches` and move below logic there.
        """
        result = []
        for query in data:
            intent = query['intent']
            query = query['data']
            words = [] # type: List[str]
            slots = [] # type: List[str]
            for part in query:
                part_words = nltk.tokenize.wordpunct_tokenize(part['text'])
                entity = part.get('entity', None)
                if entity:
                    slots.append('B-' + entity)
                    slots += ['I-' + entity] * (len(part_words) - 1)
                else:
                    slots += ['O'] * len(part_words)
                words += part_words

            result.append((words, (intent, slots)))
        return result

