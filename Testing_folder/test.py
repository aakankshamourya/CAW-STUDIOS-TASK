def reverse_string_words(s):
    w=s.split()
    w=w[::-1]
    return ' '.join(w)
print(reverse_string_words("I love JavaScript"))