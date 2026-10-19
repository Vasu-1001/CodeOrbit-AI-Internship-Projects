"""
Configuration and Knowledge Base for Task 1: Rule-Based Chatbot.
CodeOrbit Tech AI Internship - Task 1.

Defines project metadata, intent schemas, keyword rules, regex patterns,
and response banks for 26+ conversational and technical intents.
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Project Metadata
# ---------------------------------------------------------------------------
PROJECT_TITLE = "Rule-Based Conversational AI Assistant"
INTERNSHIP_TASK_NAME = "Task 1: Rule-Based Chatbot"
INTERNSHIP_ORGANIZATION = "CodeOrbit Tech"
PROJECT_VERSION = "1.0.0"
AUTHOR = "AI Intern"

# ---------------------------------------------------------------------------
# Fallback and Default Messages
# ---------------------------------------------------------------------------
FALLBACK_RESPONSES: List[str] = [
    (
        "I'm sorry, I don't quite understand that question yet. "
        "As a rule-based AI assistant, I operate on defined pattern matching rules! "
        "Try asking me about **Python programming**, **AI & Machine Learning concepts**, "
        "**Lists vs Tuples**, **OOP principles**, or type **'help'** to see everything I can answer."
    ),
    (
        "I didn't recognize that pattern in my knowledge base. "
        "Could you rephrase your question or ask about topics like **Python**, **Overfitting**, "
        "**Supervised Learning**, or **Git**? Type **'topics'** to view my supported subjects."
    ),
    (
        "That's outside my current rule dictionary! "
        "Remember, I'm a deterministic rule-based bot designed without third-party LLMs. "
        "Check out the suggested questions in the sidebar or ask **'what can you do?'**."
    ),
]

EMPTY_INPUT_RESPONSE = (
    "It looks like you submitted an empty message! "
    "Please type a question or select one of the suggested topics below."
)

WELCOME_MESSAGE = (
    "👋 Hello! I am your **CodeOrbit Tech Rule-Based Assistant**.\n\n"
    "I can answer questions regarding:\n"
    "- 🐍 **Python Programming** (features, data types, GIL, list vs tuple, OOP, venv)\n"
    "- 🤖 **Artificial Intelligence & ML** (AI vs ML, supervised learning, overfitting, deep learning)\n"
    "- 🛠️ **Software Engineering** (Git vs GitHub, Big-O notation, rule-based systems)\n"
    "- 💡 **Bot Identity & Internship FAQs**\n\n"
    "Try typing **'What is Python?'**, **'Difference between list and tuple'**, or click any suggestion below!"
)

SUGGESTED_QUESTIONS: List[str] = [
    "What is Python?",
    "Difference between list and tuple?",
    "What is the difference between AI and ML?",
    "What is overfitting and how do you prevent it?",
    "What is GIL in Python?",
    "Explain supervised vs unsupervised learning",
    "What are OOP concepts in Python?",
    "What is Big O notation?",
    "What is Git vs GitHub?",
    "Tell me a programming joke",
]

# ---------------------------------------------------------------------------
# Intent Knowledge Base
# Priority: higher value checked earlier for overlapping patterns
# ---------------------------------------------------------------------------
INTENTS: Dict[str, Dict[str, Any]] = {
    # 1. Greetings
    "greeting": {
        "priority": 20,
        "category": "Conversational",
        "keywords": [
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
            "good evening", "howdy", "sup", "what's up", "namaste"
        ],
        "patterns": [
            r"\b(hello|hi|hey|greetings|howdy)\b",
            r"\bgood\s+(morning|afternoon|evening|day)\b",
            r"\bwhat('?s|\s+is)\s+up\b",
        ],
        "responses": [
            "Hello there! How can I assist you with Python or AI concepts today?",
            "Hi! Welcome to the CodeOrbit Tech AI Assistant. What programming topic would you like to explore?",
            "Greetings! Feel free to ask me anything about Python, Machine Learning, or software engineering basics.",
        ],
    },

    # 2. Farewells
    "goodbye": {
        "priority": 20,
        "category": "Conversational",
        "keywords": ["bye", "goodbye", "see you", "farewell", "exit", "quit", "cya", "later"],
        "patterns": [
            r"\b(bye|goodbye|farewell|cya)\b",
            r"\bsee\s+you(\s+later)?\b",
            r"\btake\s+care\b",
        ],
        "responses": [
            "Goodbye! Best of luck with your coding and AI journey! Feel free to chat anytime.",
            "See you later! Keep building great projects with CodeOrbit Tech.",
            "Farewell! Remember: practice makes permanent in programming!",
        ],
    },

    # 3. Gratitude
    "thanks": {
        "priority": 20,
        "category": "Conversational",
        "keywords": ["thanks", "thank you", "thx", "appreciate it", "grateful", "awesome thanks"],
        "patterns": [
            r"\bthank(s|\s+you)\b",
            r"\bappreciate\s+it\b",
            r"\bthx\b",
        ],
        "responses": [
            "You are very welcome! Let me know if you need help with any other concept.",
            "Happy to help! Keep the great questions coming.",
            "Glad I could clarify that for you! Happy programming!",
        ],
    },

    # 4. Bot Identity
    "bot_identity": {
        "priority": 30,
        "category": "System",
        "keywords": ["who are you", "what are you", "your name", "tell me about yourself", "introduce yourself"],
        "patterns": [
            r"\bwho\s+are\s+you\b",
            r"\bwhat('?s|\s+is)\s+your\s+name\b",
            r"\btell\s+me\s+about\s+yourself\b",
            r"\bintroduce\s+yourself\b",
        ],
        "responses": [
            (
                "I am **OrbitBot**, a modular rule-based conversational AI developed for the "
                "**CodeOrbit Tech AI Internship (Task 1)**. I parse user inputs through deterministic "
                "pattern matching and regex rules without external third-party LLMs!"
            ),
        ],
    },

    # 5. Creator / Organization
    "bot_creator": {
        "priority": 30,
        "category": "System",
        "keywords": ["who created you", "who made you", "who built you", "developer", "author", "codeorbit"],
        "patterns": [
            r"\bwho\s+(created|made|built|developed|programmed)\s+you\b",
            r"\bwho\s+is\s+your\s+(creator|author|developer)\b",
            r"\bcodeorbit(\s+tech)?\b",
        ],
        "responses": [
            (
                "I was built as part of the **CodeOrbit Tech Two-Month AI Internship Program**! "
                "My architecture demonstrates modular software design, regex-based NLP preprocessing, "
                "and clean Streamlit user interfaces."
            ),
        ],
    },

    # 6. Bot Capabilities / Help
    "help_capabilities": {
        "priority": 25,
        "category": "System",
        "keywords": ["help", "what can you do", "features", "capabilities", "commands", "menu", "topics", "options"],
        "patterns": [
            r"\b(help|commands|features|capabilities|menu|topics)\b",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bhow\s+do\s+you\s+work\b",
        ],
        "responses": [
            (
                "### 🤖 What I Can Do For You:\n"
                "I am configured with **26+ intent domains** covering:\n"
                "1. **Python Core:** Definitions, features, data types, OOP principles, functions & lambdas.\n"
                "2. **Python Internals:** List vs Tuple, Global Interpreter Lock (GIL), Virtual environments (`venv`).\n"
                "3. **AI & Data Science:** AI vs ML vs DL, Supervised vs Unsupervised learning, Overfitting mitigation.\n"
                "4. **Computer Science:** Big-O Time Complexity, Git vs GitHub, Rule-based systems vs ML.\n"
                "5. **Fun & Motivation:** Tech jokes and inspirational coding quotes.\n\n"
                "*Tip: Simply type your question naturally, e.g., 'What is GIL?' or 'List vs Tuple'.*"
            ),
        ],
    },

    # 7. Bot Status / Small Talk
    "bot_status": {
        "priority": 25,
        "category": "Conversational",
        "keywords": ["how are you", "how's it going", "how do you do", "are you okay"],
        "patterns": [
            r"\bhow\s+are\s+you(\s+doing)?\b",
            r"\bhow('?s|\s+is)\s+it\s+going\b",
            r"\bhow\s+do\s+you\s+do\b",
        ],
        "responses": [
            "I'm operating at 100% CPU efficiency and zero latency! Ready to solve your programming queries.",
            "All systems operational! My pattern matchers are primed and ready. How can I help you today?",
        ],
    },

    # 8. FAQ: What is Python
    "faq_python_definition": {
        "priority": 40,
        "category": "Python",
        "keywords": ["what is python", "define python", "about python", "explain python"],
        "patterns": [
            r"\bwhat\s+is\s+python\b",
            r"\bdefine\s+python\b",
            r"\bexplain\s+python\b",
        ],
        "responses": [
            (
                "**Python** is a high-level, interpreted, general-purpose programming language created by "
                "Guido van Rossum and first released in 1991.\n\n"
                "**Key Highlights:**\n"
                "- **Readability:** Clean, indentation-based syntax resembling plain English.\n"
                "- **Multi-paradigm:** Supports Object-Oriented, Procedural, and Functional programming.\n"
                "- **Ecosystem:** Enormous standard library and rich third-party packages (NumPy, PyTorch, Streamlit, Scikit-Learn).\n"
                "- **Use Cases:** AI/ML, web backends, automation scripts, data analytics, and scientific computing."
            ),
        ],
    },

    # 9. FAQ: Python Features & Advantages
    "faq_python_features": {
        "priority": 45,
        "category": "Python",
        "keywords": ["python features", "features of python", "advantages of python", "why use python", "why is python popular"],
        "patterns": [
            r"\b(features|advantages|benefits)\s+of\s+python\b",
            r"\bwhy\s+(use|choose|learn|is)\s+python(\s+popular)?\b",
            r"\bpython\s+features\b",
        ],
        "responses": [
            (
                "**Core Advantages of Python:**\n"
                "1. **Easy to Learn & Read:** Emphasizes developer productivity with clean syntax.\n"
                "2. **Interpreted:** Executes code line-by-line, accelerating rapid prototyping and debugging.\n"
                "3. **Dynamically Typed:** Variable types are declared and checked at runtime.\n"
                "4. **Extensive Libraries:** Unmatched packages for AI (`torch`, `sklearn`), Web (`FastAPI`, `Django`), and Data (`pandas`).\n"
                "5. **Cross-Platform:** Runs seamlessly on Windows, macOS, Linux, and embedded hardware."
            ),
        ],
    },

    # 10. FAQ: List vs Tuple (Specific comparison - high priority)
    "faq_python_lists_tuples": {
        "priority": 60,
        "category": "Python",
        "keywords": [
            "list vs tuple", "difference between list and tuple", "tuple vs list",
            "lists and tuples", "compare list and tuple", "list and a tuple", "list and tuple"
        ],
        "patterns": [
            r"\b(difference\s+between\s+)?(a\s+|the\s+)?lists?\s+(vs|and|versus)\s+(a\s+|the\s+)?tuples?\b",
            r"\b(difference\s+between\s+)?(a\s+|the\s+)?tuples?\s+(vs|and|versus)\s+(a\s+|the\s+)?lists?\b",
            r"\blists?\s+(vs|versus)\s+tuples?\b",
            r"\bcompare\s+(a\s+)?lists?\s+and\s+(a\s+)?tuples?\b",
        ],
        "responses": [
            (
                "### 📋 List vs Tuple Comparison in Python:\n\n"
                "| Feature | `list` | `tuple` |\n"
                "| :--- | :--- | :--- |\n"
                "| **Mutability** | **Mutable** (items can be added, updated, or removed) | **Immutable** (cannot be altered after creation) |\n"
                "| **Syntax** | Square brackets: `[1, 2, 3]` | Parentheses: `(1, 2, 3)` |\n"
                "| **Memory & Speed** | Higher memory overhead, slightly slower | Lower memory footprint, faster iteration |\n"
                "| **Hashability** | Not hashable (cannot be a dictionary key) | Hashable (can serve as a dictionary key if items are immutable) |\n"
                "| **Typical Use** | Dynamic collections of homogeneous items | Fixed records or heterogeneous data structures |"
            ),
        ],
    },

    # 11. FAQ: Python Data Types
    "faq_python_data_types": {
        "priority": 45,
        "category": "Python",
        "keywords": ["python data types", "data types in python", "built in data types", "primitive data types"],
        "patterns": [
            r"\b(python\s+)?data\s+types(\s+in\s+python)?\b",
            r"\btypes\s+of\s+data\s+in\s+python\b",
        ],
        "responses": [
            (
                "**Built-in Data Types in Python:**\n"
                "- **Numeric:** `int` (integers), `float` (decimals), `complex` (e.g. `2+3j`)\n"
                "- **Sequence:** `str` (text), `list` (mutable array), `tuple` (immutable sequence), `range`\n"
                "- **Mapping:** `dict` (key-value store, hash table)\n"
                "- **Set Types:** `set` (unique, unordered items), `frozenset` (immutable set)\n"
                "- **Boolean:** `bool` (`True` or `False`)\n"
                "- **Binary:** `bytes`, `bytearray`, `memoryview`\n"
                "- **None Type:** `NoneType` (`None` representing absence of value)"
            ),
        ],
    },

    # 12. FAQ: Global Interpreter Lock (GIL)
    "faq_python_gil": {
        "priority": 60,
        "category": "Python",
        "keywords": ["gil", "what is gil", "global interpreter lock", "explain gil", "python gil"],
        "patterns": [
            r"\bwhat\s+is\s+(the\s+)?gil\b",
            r"\bglobal\s+interpreter\s+lock\b",
            r"\bexplain\s+(the\s+)?gil\b",
        ],
        "responses": [
            (
                "**The Global Interpreter Lock (GIL):**\n\n"
                "The GIL is a mutex (mutual exclusion lock) used by CPython (the standard Python implementation) "
                "that allows only **one native thread to execute Python bytecode at any given moment**.\n\n"
                "- **Why it exists:** CPython's memory management uses reference counting, which is not natively thread-safe. "
                "The GIL prevents race conditions during memory deallocation.\n"
                "- **Impact:** Pure CPU-bound Python threads cannot run in true parallel on multiple CPU cores.\n"
                "- **Workarounds:** Use the `multiprocessing` module (separate processes with separate memory), "
                "C-extensions (like NumPy/PyTorch which release the GIL during heavy computation), or the newer free-threaded Python 3.13+."
            ),
        ],
    },

    # 13. FAQ: Virtual Environment
    "faq_python_virtualenv": {
        "priority": 50,
        "category": "Python",
        "keywords": ["virtual environment", "venv", "what is venv", "why use venv", "python venv"],
        "patterns": [
            r"\bwhat\s+is\s+(a\s+)?(venv|virtual\s+environment)\b",
            r"\bwhy\s+use\s+(a\s+)?(venv|virtual\s+environment)\b",
            r"\bhow\s+to\s+create\s+(a\s+)?venv\b",
        ],
        "responses": [
            (
                "**Python Virtual Environments (`venv`):**\n\n"
                "A virtual environment is an isolated directory tree that contains a specific Python installation "
                "and an independent set of packages.\n\n"
                "- **Why use it?** It prevents dependency conflicts between projects (e.g., Project A needing pandas 1.5 "
                "while Project B needs pandas 2.2).\n"
                "- **How to create one in Windows PowerShell:**\n"
                "  ```powershell\n"
                "  python -m venv myenv\n"
                "  .\\myenv\\Scripts\\Activate.ps1\n"
                "  pip install -r requirements.txt\n"
                "  ```"
            ),
        ],
    },

    # 14. FAQ: Functions and Lambdas
    "faq_python_lambda": {
        "priority": 50,
        "category": "Python",
        "keywords": ["lambda function", "what is lambda", "anonymous function", "lambda vs def"],
        "patterns": [
            r"\bwhat\s+is\s+(a\s+)?lambda(\s+function)?\b",
            r"\banonymous\s+functions?\b",
            r"\blambda\s+vs\s+def\b",
        ],
        "responses": [
            (
                "**Lambda Functions in Python:**\n\n"
                "A `lambda` function is a small, anonymous function that can have any number of arguments, "
                "but only a **single expression** whose result is automatically returned.\n\n"
                "**Syntax:** `lambda arguments: expression`\n\n"
                "**Example:**\n"
                "```python\n"
                "# Standard function\n"
                "def square(x):\n"
                "    return x ** 2\n\n"
                "# Equivalent Lambda\n"
                "square_lambda = lambda x: x ** 2\n"
                "```\n"
                "**Common Use:** Short callbacks passed into higher-order functions like `map()`, `filter()`, or `sorted(..., key=...)`."
            ),
        ],
    },

    # 15. FAQ: OOP Concepts
    "faq_python_oop": {
        "priority": 50,
        "category": "Python",
        "keywords": ["oop concepts", "oop in python", "object oriented programming", "classes and objects", "inheritance polymorphism"],
        "patterns": [
            r"\boop(\s+concepts|\s+in\s+python)?\b",
            r"\bobject\s+oriented\s+programming\b",
            r"\bclasses\s+and\s+objects\b",
            r"\bexplain\s+oop\b",
        ],
        "responses": [
            (
                "**The 4 Pillars of Object-Oriented Programming (OOP) in Python:**\n\n"
                "1. **Encapsulation:** Bundling data (attributes) and methods that operate on that data into classes, "
                "protecting internal state using naming conventions (e.g., `_protected`, `__private`).\n"
                "2. **Inheritance:** Enabling a child class to inherit attributes and methods from a parent class (`class Child(Parent):`).\n"
                "3. **Polymorphism:** Allowing different classes to implement the same method interface (e.g., method overriding or duck typing).\n"
                "4. **Abstraction:** Hiding complex implementation details and showing only the relevant interface using the `abc` module (`@abstractmethod`)."
            ),
        ],
    },

    # 16. FAQ: What is AI
    "faq_ai_definition": {
        "priority": 40,
        "category": "AI & ML",
        "keywords": ["what is ai", "define ai", "artificial intelligence definition", "explain artificial intelligence"],
        "patterns": [
            r"\bwhat\s+is\s+(ai|artificial\s+intelligence)\b",
            r"\bdefine\s+(ai|artificial\s+intelligence)\b",
            r"\bexplain\s+(ai|artificial\s+intelligence)\b",
        ],
        "responses": [
            (
                "**Artificial Intelligence (AI)** is a broad branch of computer science dedicated to creating "
                "systems capable of performing tasks that traditionally require human intelligence.\n\n"
                "These tasks include:\n"
                "- Reasoning, deduction, and decision-making (e.g. Minimax search in Tic-Tac-Toe)\n"
                "- Visual perception and image classification (e.g. MobileNet, CNNs)\n"
                "- Speech recognition and Natural Language Processing (NLP)\n"
                "- Automated problem-solving and predictive modeling."
            ),
        ],
    },

    # 17. FAQ: AI vs ML vs DL
    "faq_ai_vs_ml": {
        "priority": 55,
        "category": "AI & ML",
        "keywords": ["difference between ai and ml", "ai vs ml", "ai vs machine learning", "ai vs ml vs dl"],
        "patterns": [
            r"\b(difference\s+between\s+)?(ai\s+(vs|and|versus)\s+(ml|machine\s+learning))\b",
            r"\bai\s+vs\s+ml(\s+vs\s+dl)?\b",
            r"\bcompare\s+ai\s+and\s+machine\s+learning\b",
        ],
        "responses": [
            (
                "### 🎯 AI vs Machine Learning vs Deep Learning:\n\n"
                "1. **Artificial Intelligence (AI):** The overarching umbrella term for machines demonstrating cognitive capability. "
                "Includes both symbolic/rule-based systems and statistical learners.\n"
                "2. **Machine Learning (ML):** A subfield of AI where algorithms learn statistical patterns from historical data "
                "to make predictions without explicit step-by-step rules (e.g., Naive Bayes, Random Forests, Linear Regression).\n"
                "3. **Deep Learning (DL):** A specialized subset of ML based on multi-layered Artificial Neural Networks "
                "(CNNs, Transformers) capable of learning hierarchical feature representations from raw high-dimensional data."
            ),
        ],
    },

    # 18. FAQ: Supervised vs Unsupervised
    "faq_supervised_unsupervised": {
        "priority": 55,
        "category": "AI & ML",
        "keywords": ["supervised vs unsupervised", "supervised learning", "unsupervised learning", "difference between supervised and unsupervised"],
        "patterns": [
            r"\bsupervised\s+(vs|and|versus)\s+unsupervised\b",
            r"\bwhat\s+is\s+supervised\s+learning\b",
            r"\bwhat\s+is\s+unsupervised\s+learning\b",
        ],
        "responses": [
            (
                "### 🔍 Supervised vs Unsupervised Learning:\n\n"
                "- **Supervised Learning:**\n"
                "  - **Input:** Labeled training data $(X, y)$.\n"
                "  - **Goal:** Learn a mapping function $f(X) \\to y$ to predict labels on unseen data.\n"
                "  - **Examples:** Spam detection (Spam/Ham), Image classification (Dog/Cat), House price prediction.\n"
                "- **Unsupervised Learning:**\n"
                "  - **Input:** Unlabeled data $(X)$ with no ground-truth targets.\n"
                "  - **Goal:** Discover latent structures, groupings, or compressed representations.\n"
                "  - **Examples:** Customer segmentation (K-Means), Dimensionality reduction (PCA), Anomaly detection."
            ),
        ],
    },

    # 19. FAQ: Overfitting
    "faq_overfitting": {
        "priority": 55,
        "category": "AI & ML",
        "keywords": ["what is overfitting", "overfitting", "how to prevent overfitting", "bias variance tradeoff", "underfitting"],
        "patterns": [
            r"\bwhat\s+is\s+overfitting\b",
            r"\bhow\s+to\s+prevent\s+overfitting\b",
            r"\boverfitting\s+vs\s+underfitting\b",
            r"\bbias\s+variance\s+tradeoff\b",
        ],
        "responses": [
            (
                "**Overfitting in Machine Learning:**\n\n"
                "Overfitting occurs when a statistical or machine learning model learns the training data **too well**, "
                "memorizing noise and random fluctuations rather than the underlying general trend. "
                "Result: near-100% training accuracy but poor generalization on validation/test sets.\n\n"
                "**How to Prevent Overfitting:**\n"
                "1. **Regularization:** Add L1 (Lasso) or L2 (Ridge) penalties to weights.\n"
                "2. **Cross-Validation:** Use K-Fold cross validation to evaluate out-of-fold generalization.\n"
                "3. **More Data & Augmentation:** Expand the dataset or apply synthetic augmentation.\n"
                "4. **Pruning & Dropout:** Drop neurons randomly during training in deep networks or prune decision trees.\n"
                "5. **Early Stopping:** Halt model training when validation loss stops decreasing."
            ),
        ],
    },

    # 20. FAQ: Deep Learning
    "faq_deep_learning": {
        "priority": 50,
        "category": "AI & ML",
        "keywords": ["deep learning", "what is deep learning", "neural networks", "how does deep learning work"],
        "patterns": [
            r"\bwhat\s+is\s+deep\s+learning\b",
            r"\bexplain\s+deep\s+learning\b",
            r"\bwhat\s+are\s+neural\s+networks\b",
        ],
        "responses": [
            (
                "**Deep Learning:**\n\n"
                "Deep Learning is a subset of Machine Learning powered by Artificial Neural Networks with multiple hidden layers "
                "(hence 'deep').\n\n"
                "- **Key Feature:** Eliminates manual feature engineering. The network automatically extracts low-level "
                "(edges, textures) up to high-level semantic features (faces, objects).\n"
                "- **Key Architectures:**\n"
                "  - **CNNs:** Convolutional Neural Networks (vision, image classification)\n"
                "  - **RNNs/LSTMs:** Recurrent Networks (time series, sequential audio)\n"
                "  - **Transformers:** Attention-based networks powering modern LLMs (GPT, Gemini, BERT)."
            ),
        ],
    },

    # 21. FAQ: Rule-Based Systems vs Machine Learning
    "faq_rule_based_vs_ml": {
        "priority": 60,
        "category": "Computer Science",
        "keywords": ["rule based vs ml", "rule based system", "rule based chatbot vs ai", "expert system", "symbolic ai"],
        "patterns": [
            r"\brule\s+based(\s+chatbot|\s+system)?\s+(vs|and|versus)\s+(ml|machine\s+learning|ai)\b",
            r"\bwhat\s+is\s+a\s+rule\s+based\s+system\b",
            r"\bexpert\s+systems?\b",
        ],
        "responses": [
            (
                "### ⚖️ Rule-Based Systems vs Machine Learning:\n\n"
                "| Aspect | Rule-Based Systems (Deterministic) | Machine Learning (Probabilistic) |\n"
                "| :--- | :--- | :--- |\n"
                "| **Mechanism** | Handcrafted `if/elif` logic, regex, decision trees | Statistical parameters learned from training data |\n"
                "| **Predictability** | 100% deterministic, zero hallucinations, fully auditable | Probabilistic, flexible, can hallucinate or drift |\n"
                "| **Data Needs** | Needs domain expert rules, zero training datasets | Requires substantial clean, labeled training data |\n"
                "| **Maintenance** | Rule sets grow complex with scaling | Requires model retraining, data pipelines, and MLOps |\n"
                "| **Example** | This Task 1 Chatbot, tax calculators, medical diagnostic trees | Task 6 Spam Classifier, Task 3 Vision Studio |"
            ),
        ],
    },

    # 22. FAQ: Big-O Time Complexity
    "faq_time_complexity": {
        "priority": 50,
        "category": "Computer Science",
        "keywords": ["big o", "time complexity", "big o notation", "space complexity", "algorithmic efficiency"],
        "patterns": [
            r"\bwhat\s+is\s+(big\s+o|time\s+complexity)\b",
            r"\bexplain\s+big\s+o(\s+notation)?\b",
            r"\bspace\s+complexity\b",
        ],
        "responses": [
            (
                "**Big-O Notation and Time Complexity:**\n\n"
                "Big-O notation describes the upper bound of the execution time or memory space required by an algorithm "
                "as the input size ($n$) scales towards infinity.\n\n"
                "- $O(1)$ — **Constant Time:** Hash table lookup, list indexing by index.\n"
                "- $O(\\log n)$ — **Logarithmic Time:** Binary search on sorted arrays.\n"
                "- $O(n)$ — **Linear Time:** Single iteration over an array of size $n$.\n"
                "- $O(n \\log n)$ — **Linearithmic Time:** Efficient sorting algorithms (Merge Sort, Timsort).\n"
                "- $O(n^2)$ — **Quadratic Time:** Nested loops (Bubble Sort, brute-force comparisons).\n"
                "- $O(2^n)$ / $O(n!)$ — **Exponential / Factorial:** Recursive exhaustive search without memoization."
            ),
        ],
    },

    # 23. FAQ: Git vs GitHub
    "faq_git_github": {
        "priority": 55,
        "category": "Computer Science",
        "keywords": ["git vs github", "difference between git and github", "what is git", "what is github"],
        "patterns": [
            r"\bgit\s+(vs|and|versus)\s+github\b",
            r"\bwhat\s+is\s+git\b",
            r"\bwhat\s+is\s+github\b",
            r"\bdifference\s+between\s+git\s+and\s+github\b",
        ],
        "responses": [
            (
                "### 🐙 Git vs GitHub:\n\n"
                "- **Git:** A free, open-source **Distributed Version Control System (DVCS)** installed locally on your machine. "
                "It tracks file changes, manages commits, and handles branching/merging.\n"
                "- **GitHub:** A cloud-based **hosting service for Git repositories**. "
                "It provides a web GUI, collaboration tools, Pull Requests, Issue tracking, CI/CD pipelines (GitHub Actions), and user permissions.\n\n"
                "*Analogy: Git is like the local camera; GitHub is like the online photo gallery where you share and collaborate.*"
            ),
        ],
    },

    # 24. Humor / Joke
    "joke": {
        "priority": 25,
        "category": "Fun",
        "keywords": ["joke", "tell me a joke", "programming joke", "say something funny", "make me laugh"],
        "patterns": [
            r"\b(tell\s+me\s+a\s+)?joke\b",
            r"\bsomething\s+funny\b",
            r"\bmake\s+me\s+laugh\b",
        ],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "There are only 10 types of people in the world: those who understand binary, and those who don't! 😄",
            "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?' 🍻",
            "Why did the Python programmer get lost in the woods? They couldn't find their indentation! 🐍",
        ],
    },

    # 25. Inspirational Coding Quote
    "quote": {
        "priority": 25,
        "category": "Fun",
        "keywords": ["quote", "inspire me", "motivational quote", "coding quote", "give me a quote"],
        "patterns": [
            r"\b(inspirational\s+|motivational\s+|coding\s+)?quote\b",
            r"\binspire\s+me\b",
            r"\bgive\s+me\s+a\s+quote\b",
        ],
        "responses": [
            "\"First, solve the problem. Then, write the code.\" — John Johnson 💻",
            "\"Simplicity is prerequisite for reliability.\" — Edsger W. Dijkstra ✨",
            "\"Talk is cheap. Show me the code.\" — Linus Torvalds 🚀",
            "\"The only way to learn a new programming language is by writing programs in it.\" — Dennis Ritchie 📚",
        ],
    },

    # 26. Internship Summary
    "internship_summary": {
        "priority": 35,
        "category": "System",
        "keywords": ["internship tasks", "codeorbit internship", "tasks completed", "projects list", "portfolio"],
        "patterns": [
            r"\b(internship\s+tasks|portfolio|projects\s+list)\b",
            r"\bwhat\s+projects\s+are\s+in\s+this\s+internship\b",
        ],
        "responses": [
            (
                "**CodeOrbit Tech Two-Month AI Internship Portfolio:**\n\n"
                "1. **Task 1: Rule-Based Chatbot** (Deterministic NLP, regex intents, Streamlit)\n"
                "2. **Task 2: Tic-Tac-Toe AI** (Adversarial game-tree search via Minimax)\n"
                "3. **Task 3: AI Image Classification Studio** (PyTorch vision backbones, ImageNet)\n"
                "4. **Task 4: Smart Movie Recommender** (Content-based filtering, TF-IDF, Cosine Similarity)\n"
                "5. **Task 5: Customer Sentiment Analysis** (VADER polarity heuristics, 3-class distribution)\n"
                "6. **Task 6: AI Spam Email Detector** (Multinomial Naive Bayes, zero data leakage)"
            ),
        ],
    },
}
