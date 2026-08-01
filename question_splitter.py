import re

def split_into_questions(full_document_text: str) -> list[dict]:
    """
    Splits a worksheet's full document text into individual numbered questions.
    
    Why this is separate from vector-based chunking:
    Vector-based chunking splits a document into fixed-size overlapping segments to create 
    small, indexable vectors for similarity search. While this is great for answering a single 
    specific question, it destroys the structure of worksheet questions (e.g., cutting 
    a question or its options in half).
    
    This splitter, on the other hand, preserves the natural boundaries of each question 
    by detecting numbering patterns (like '1.', 'Q2.', 'Question 3:') and extracting the 
    entire question block intact. This is only used for the 'Solve Entire Document' feature, 
    ensuring that we pass complete, contiguous question contexts to the LLM.
    """
    if not full_document_text:
        return []
        
    # Pattern to detect numbered questions:
    # - (?:^|\n): Start of the text or start of a new line
    # - \s*: Optional leading whitespace
    # - (?:[Qq]uestion|[Qq])?\s*: Optional "Question" or "Q" (case-insensitive) followed by optional spaces
    # - (\d+): One or more digits (captured as group 1)
    # - [\.\):-]: Punctuation separating number from question (e.g., '.', ')', ':', '-')
    # - \s+: Required whitespace following the separator
    pattern = re.compile(r'(?:^|\n)\s*(?:[Qq]uestion|[Qq])?\s*(\d+)[\.\):-]\s+')
    
    matches = list(pattern.finditer(full_document_text))
    if not matches:
        return []
        
    questions = []
    for i, match in enumerate(matches):
        q_num = int(match.group(1))
        start_idx = match.start()
        
        # Determine where this question block ends (start of the next question, or end of document)
        if i + 1 < len(matches):
            end_idx = matches[i+1].start()
        else:
            end_idx = len(full_document_text)
            
        q_text = full_document_text[start_idx:end_idx].strip()
        questions.append({
            "number": q_num,
            "question_text": q_text
        })
        
    return questions
