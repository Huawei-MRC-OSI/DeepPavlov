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
from pathlib import Path

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.models.lr_scheduled_model import LRScheduledModel
from deeppavlov.core.common.registry import register

from typing import List,Any

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import models

@register('intent_slot_classifier')
class IntentSlotClassifier(LRScheduledModel):
    def __init__(self, n_intents=7, n_slots=70, vocab_size=5000, save_path=None, *args, **kwargs):
        self.n_intents = n_intents
        self.n_slots = n_slots
        self.vocab_size = vocab_size
        self.save_path = save_path
        self.load()
        self.model.summary()

    def train_on_batch(self, x:List[Any], y_intent_ids:List[Any], y_slot_ids:List[Any]):
        x = pad_sequences(x)
        y_slot_ids = pad_sequences(y_slot_ids)
        self.model.train_on_batch(x, [y_intent_ids, y_slot_ids])

    def infer_on_batch(self, x:List[Any], y_intent_ids:List[Any]=None, y_slot_ids:List[Any]=None):
        x = pad_sequences(x)
        if y_intent_ids:
            y_slot_ids = pad_sequences(y_slot_ids)
            metrics_values = self.model.test_on_batch(x, [y_intent_ids, y_slot_ids])
            return metrics_values
        else:
            predictions = self.model.predict(x)
            return predictions

    def process_event(self, event_name, data):
        pass

    def save(self):
        if self.save_path:
            model.save(save_path)
        pass

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
        preds = self.infer_on_batch(data)
        preds = [np.array(preds[0], dtype="float64").tolist(), np.array(preds[1], dtype="float64").tolist()]
        return preds

    def load(self):
        if self.save_path and Path(self.save_path).exists():
            model = models.load_model(self.save_path)
        else:
            model = self.get_model()
        self.model = model

    def get_model(self):
        inputs = layers.Input(shape=(None,))
        embed = layers.Embedding(self.vocab_size, 18, mask_zero=True)(inputs)
        rnn_intents = layers.Bidirectional(layers.RNN(layers.GRUCell(256)))(embed)
        result_intents = layers.Dense(self.n_intents, activation='softmax')(rnn_intents)
        rnn_slots = layers.Bidirectional(layers.RNN(layers.GRUCell(self.n_slots), return_sequences=True), merge_mode='sum')(embed)
        result_slots = layers.Activation('softmax')(rnn_slots)
        model = models.Model(inputs=[inputs], outputs=[result_intents, result_slots])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        return model

