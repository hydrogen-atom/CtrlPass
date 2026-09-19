"""Offline tests for grounded material answers and knowledge graphs."""
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.learning_provider import LearningProvider


MATERIAL = {
    'chunk_count': 2,
    'knowledge_points_json': '[{"id":1,"name":"二分查找"}]',
}
SOURCES = [
    {'chunk_id': 1, 'page': None, 'text': '二分查找要求数据有序。'},
    {'chunk_id': 2, 'page': 3, 'text': '每轮排除一半候选范围。'},
]


def test_answer_is_grounded_to_retrieved_chunks():
    provider = LearningProvider()
    with patch.object(provider, 'retrieve_sources', return_value=SOURCES), patch.object(provider, 'ask', return_value={
        'answer': '二分查找要求数据有序。[片段1]',
        'source_refs': [{'chunk_id': 1, 'page': 999}],
    }):
        result = provider.answer(MATERIAL, '二分查找有什么前提？', Path('.'))
    assert result['answer'].startswith('二分查找')
    # The server uses trusted retrieval metadata instead of the model-provided page.
    assert result['source_refs'] == [{'chunk_id': 1, 'page': None}]


def test_answer_rejects_reference_not_in_retrieval_result():
    provider = LearningProvider()
    with patch.object(provider, 'retrieve_sources', return_value=SOURCES), patch.object(provider, 'ask', return_value={
        'answer': '无依据回答',
        'source_refs': [{'chunk_id': 99, 'page': None}],
    }):
        with pytest.raises(ValueError, match='Ungrounded'):
            provider.answer(MATERIAL, '问题', Path('.'))


def test_knowledge_graph_validates_nodes_edges_and_sources():
    provider = LearningProvider()
    model_graph = {
        'nodes': [
            {'id': 'search', 'label': '二分查找', 'category': '算法', 'description': '缩小范围'},
            {'id': 'sorted', 'label': '有序数据', 'category': '前提', 'description': '保持有序'},
        ],
        'edges': [
            {'source': 'search', 'target': 'sorted', 'relation': '要求', 'evidence': '资料明确说明'},
        ],
        'source_refs': [{'chunk_id': 1, 'page': None}],
    }
    with patch.object(provider, 'retrieve_sources', return_value=SOURCES), patch.object(provider, 'ask', return_value=model_graph):
        graph = provider.knowledge_graph(MATERIAL, '算法关系', Path('.'))
    assert graph['nodes'][0]['id'] == 'search'
    assert graph['edges'][0]['target'] == 'sorted'
    assert graph['source_refs'] == [{'chunk_id': 1, 'page': None}]
