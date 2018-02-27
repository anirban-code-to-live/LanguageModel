echo 'Installing nltk..'
pip3 install nltk --user
echo 'nltk installed'

echo 'Generating sentence'

export PYTHONPATH='.'
python3 src/main.py

echo 'Sentence generated'
echo 'Look at sentence.txt file'