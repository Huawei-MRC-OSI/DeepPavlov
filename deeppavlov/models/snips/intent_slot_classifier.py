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

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import models

@register('intent_slot_classifier')
class IntentSlotClassifier(LRScheduledModel):
    def __init__(self, n_intents=7, n_slots=70, vocab_size=5000, *args, **kwargs):
        inputs = layers.Input(shape=(None,))
        embed = layers.Embedding(vocab_size, 18, mask_zero=True)(inputs)
        rnn_intents = layers.Bidirectional(layers.RNN(layers.GRUCell(256)))(embed)
        result_intents = layers.Dense(n_intents, activation='softmax')(rnn_intents)
        rnn_slots = layers.Bidirectional(layers.RNN(layers.GRUCell(n_slots), return_sequences=True), merge_mode='sum')(embed)
        result_slots = layers.Activation('softmax')(rnn_slots)
        model = models.Model(inputs=[inputs], outputs=[result_intents, result_slots])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
        self.model = model

    def train_on_batch(self, x:List[Any], y_intent_ids:List[Any], y_slot_ids:List[Any]):
        x = pad_sequences(x)
        y_slot_ids = pad_sequences(y_slot_ids)
        self.model.train_on_batch(x, [y_intent_ids, y_slot_ids])
        print("batch")

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

