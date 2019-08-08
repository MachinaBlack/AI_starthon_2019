#nsml: scatterlab/nsml-custom:cuda9.2-cudnn7-torch1.1-mecab
from distutils.core import setup
setup(
        name='18_tcls_query_baseline',
        version='1.0',
        description='18_tcls_query_baseline',
        install_requires=[
            'soynlp',
            'konlpy',
            'sentencepiece'
        ]
)
