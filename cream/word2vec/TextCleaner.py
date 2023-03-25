class cleaner:
    def clean(string:str) -> str:
        '''clean text'''

        listed = [c for c in string.splitlines() if c != '']
        joined = ' '.join(listed)

        return joined