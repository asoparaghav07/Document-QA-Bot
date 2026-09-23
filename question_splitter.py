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
    # - Group 1: Entire prefix including optional question keyword and number
    # - Group 2: The question digits
    pattern = re.compile(r'(?:^|\n)\s*((?:[Qq]uestion|[Qq])?\s*(\d+)[\.\):-]\s+)')
    
    matches = list(pattern.finditer(full_document_text))
    if not matches:
        return []
        
    # List of common worksheet-style imperative verbs
    imperative_pattern = re.compile(
        r'^\s*(?:Name|List|Describe|Explain|Identify|Define|Calculate|State|Give|Compare)\b',
        re.IGNORECASE
    )

    questions = []
    for i, match in enumerate(matches):
        matched_prefix = match.group(1)
        q_num = int(match.group(2))
        has_keyword = bool(re.search(r'^(?:[Qq]uestion|[Qq])\b', matched_prefix.strip(), re.IGNORECASE))
        start_idx = match.start()
        prefix_end_idx = match.end()
        
        # Determine where this question block ends (start of the next question, or end of document)
        if i + 1 < len(matches):
            end_idx = matches[i+1].start()
        else:
            end_idx = len(full_document_text)
            
        q_text = full_document_text[start_idx:end_idx].strip()
        after_prefix_text = full_document_text[prefix_end_idx:end_idx].strip()
        has_imperative = bool(imperative_pattern.search(after_prefix_text))
        
        # False-positive filtering:
        # Require an explicit question keyword (e.g. "Question 1", "Q1"),
        # an imperative verb (e.g. "Name", "Explain", "Calculate"),
        # OR a question mark '?' in the text.
        # This prevents standard section headers (e.g. "1. Executive Summary") from being misidentified.
        if has_keyword or has_imperative or "?" in q_text:
            final_q_text = after_prefix_text if after_prefix_text else q_text
            if "?" in final_q_text:
                final_q_text = final_q_text[:final_q_text.find("?") + 1].strip()
            questions.append({
                "number": q_num,
                "question_text": final_q_text
            })
        
    return questions
