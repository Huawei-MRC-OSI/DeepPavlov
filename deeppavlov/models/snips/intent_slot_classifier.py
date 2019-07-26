# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import numpy as np

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.models.lr_scheduled_model import LRScheduledModel
from deeppavlov.core.common.registry import register

from typing import List,Any

@register('intent_slot_classifier')
class IntentSlotClassifier(LRScheduledModel):
    def __init__(self, *args, **kwargs):
        print('intent_slot_classifier created')

    def train_on_batch(self, x:List[Any], y_intent_ids:List[Any], y_slot_ids:List[Any]):
        print('train_on_batch called')
        print('===> X', 'len', len(x), 'data', x)
        print('===> YI', 'len', len(y_intent_ids), 'data', y_intent_ids)
        print('===> YS', 'len', len(y_slot_ids), 'data', y_slot_ids)
        pass

    def process_event(self, event_name, data):
        print('process_event called, event', event_name)
        pass

    def save(self):
        print('save called')

    def __call__(self, data: List[List[np.ndarray]], *args) -> List[List[float]]:
        """
        Infer on the given data

        Args:
            data: list of tokenized text samples
            *args: additional arguments

        Returns:
            for each sentence:
                vector of probabilities to belong with each class
                or list of labels sentence belongs with
        """
        # preds = np.array(self.infer_on_batch(data), dtype="float64").tolist()
        print('__call__ called')
        return list(map(lambda words: [0.0], data))

