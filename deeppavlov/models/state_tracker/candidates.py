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

import itertools
import operator
from overrides import overrides
from typing import Any, Union, List, Dict, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


class SlotValueFilter:
    """
    Inputs lists of (slot, it's value) pairs in descending by score order, combines them
    keeping the order, and outputs top n candidates for each slot.

    Example:
        >>> f = SlotValueFilter(max_num_values=2)

        >>> f([('eyes', 'blue'), ('eyes', 'green'), ('eyes', 'red')])
        [('eyes', 'blue'), ('eyes', 'green')]

        >>> f([('eyes', 'blue'), ('eyes', 'green')], [('hair', 'long'), ('eyes', 'red')])
        [('eyes', 'blue'), ('eyes', 'green'), ('hair', 'long')]

        >>> f([('eyes', 'blue'), ('eyes', 'green')], {'hair': 'long', 'eyes': 'red'})
        [('eyes', 'blue'), ('eyes', 'green'), ('hair', 'long')

    Parameters:
        max_num_values: maximum number of values for each slot.
    """

    def __init__(self, max_num_values: int = None, **kwargs):
        self.max_num = max_num_values

    def __call__(self, *batch: List[Union[List[Tuple[str, Any]], Dict[str, Any]]])\
            -> List[Tuple[str, Any]]:
        # slot value pairs as generator of tuples
        pairs = itertools.chain(*map(self._dict2list, batch))
        pairs_filtered = []
        _slot = operator.itemgetter(0)
        for slot, slot_pairs in itertools.groupby(sorted(pairs, key=_slot), _slot):
            uniq_pairs = map(operator.itemgetter(0), itertools.groupby(slot_pairs))
            pairs_filtered.extend(p for _, p in zip(range(self.max_num), uniq_pairs))
        return pairs_filtered

    @staticmethod
    def _dict2list(x):
        if isinstance(x, Dict):
            return x.items()
        return x


@register("slot_value_filter")
class SlotValueFilterComponent(Component):
    """
    The component wraps
    :class:`~deeppavlov.models.state_tracker.candidates.SlotValueFilter`
    into DeepPavlov abstract class :class:`~deeppavlov.models.component.Component`.

    It takes batches of samples and feeds them to
    :method:`~deeppavlov.models.state_tracker.candidates.SlotValueFilter.__call__`.

    Example:
        >>> fc = c.SlotValueFilterComponent(max_num=2)
        >>> v1 = [[('s1', 1), ('s1', 2), ('s1', 3)], [('eyes', 'blue'), ('eyes', 'green')]]
        >>> v2 = [{'s1': 4}, {'hair': 'long', 'eyes': 'red'}]
        >>> v3 = [{}, {}]

        >>> fc(v1, v2, v3)
        [[('s1', 1), ('s1', 2)],
         [('eyes', 'blue'), ('eyes', 'green'), ('hair', 'long')]]

    Parameters:
        max_num_values: maximum number of values for each slot.
    """

    def __init__(self, max_num_values: int = None, **kwargs) -> None:
        self.filter = SlotValueFilter(max_num_values)

    @overrides
    def __call__(self, *batch: List[List[Any]]) -> List[List[Any]]:
        # batch is a list of batches for different variables
        return [self.filter(*sample) for sample in zip(*batch)]
