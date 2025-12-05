/**
 * DATA DEFINITIONS
 */
const CURRICULUM = [
  {
    id: 'stage-1',
    title: 'STAGE 1 – FOUNDATIONS',
    modules: [
      {
        id: 'mod-1-1',
        title: 'Programming Foundations',
        subtopics: [
          { 
            id: 't-1-1-1', 
            title: 'Python Syntax Essentials', 
            points: ['Scalar Types (int, float, bool)', 'Control Flow (if, for, while)', 'Functions & Lambda Expressions', 'Modules & Package Management'],
            resources: [{ title: 'Official Python 3 Tutorial', url: 'https://docs.python.org/3/tutorial/introduction.html' }, { title: 'Real Python: Formatting & Syntax', url: 'https://realpython.com/python-first-steps/' }]
          },
          { 
            id: 't-1-1-2', 
            title: 'Data Structures & Algorithms', 
            points: ['Lists, Sets & Dictionaries', 'Stacks & Queues', 'Time Complexity (Big O)', 'Hash Maps'],
            resources: [{ title: 'Problem Solving with Algorithms', url: 'https://runestone.academy/ns/books/published/pythonds/index.html' }, { title: 'Big O Notation Explained', url: 'https://www.youtube.com/watch?v=v4cd1O4zkGw' }]
          },
          { 
            id: 't-1-1-3', 
            title: 'Object-Oriented Design', 
            points: ['Class Definition & Objects', 'Inheritance & Composition', 'Magic Methods (__init__)', 'Polymorphism Patterns'],
            resources: [{ title: 'OOP in Python (Real Python)', url: 'https://realpython.com/python3-object-oriented-programming/' }, { title: 'Corey Schafer: OOP Tutorial', url: 'https://www.youtube.com/watch?v=ZDa-Z5JzLYM' }]
          },
        ],
      },
      {
        id: 'mod-1-2',
        title: 'Mathematics for ML',
        subtopics: [
          { 
            id: 't-1-2-1', 
            title: 'Linear Algebra Fundamentals', 
            points: ['Vectors & Matrices', 'Dot Product & Matrix Multiplication', 'Eigenvalues & Eigenvectors', 'Matrix Decomposition'],
            resources: [{ title: '3Blue1Brown: Essence of Linear Algebra', url: 'https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab' }, { title: 'Immersive Linear Algebra', url: 'http://immersivemath.com/ila/index.html' }]
          },
          { 
            id: 't-1-2-2', 
            title: 'Multivariate Calculus', 
            points: ['Derivatives & Partial Derivatives', 'The Chain Rule', 'Gradients & Hessians', 'Jacobian Matrices'],
            resources: [{ title: 'Khan Academy: Multivariable Calculus', url: 'https://www.khanacademy.org/math/multivariable-calculus' }, { title: 'The Matrix Calculus You Need', url: 'https://arxiv.org/abs/1802.01528' }]
          },
          { 
            id: 't-1-2-3', 
            title: 'Probability & Statistics', 
            points: ['Probability Distributions', 'Bayes Theorem', 'Hypothesis Testing', 'Statistical Significance'],
            resources: [{ title: 'Seeing Theory: Visual Statistics', url: 'https://seeing-theory.brown.edu/' }, { title: 'StatQuest: Probability Basics', url: 'https://www.youtube.com/watch?v=oHWbNlP8G4U' }]
          },
        ],
      },
    ],
  },
  {
    id: 'stage-2',
    title: 'STAGE 2 – DATA ENGINEERING',
    modules: [
      {
        id: 'mod-2-1',
        title: 'Data Processing',
        subtopics: [
          { 
            id: 't-2-1-1', 
            title: 'Data Manipulation Libraries', 
            points: ['Pandas DataFrames & Series', 'NumPy Broadcasting', 'Indexing & Slicing', 'Vectorized Operations'],
            resources: [{ title: '10 Minutes to pandas', url: 'https://pandas.pydata.org/docs/user_guide/10min.html' }, { title: 'NumPy Visual Guide', url: 'https://jalammar.github.io/visual-numpy/' }]
          },
          { 
            id: 't-2-1-2', 
            title: 'SQL for Data Science', 
            points: ['SELECT, FROM, WHERE clauses', 'Joins (Inner, Left, Right)', 'Group By & Aggregations', 'Window Functions'],
            resources: [{ title: 'Mode Analytics: SQL Tutorial', url: 'https://mode.com/sql-tutorial/' }, { title: 'SQLZoo Practice', url: 'https://sqlzoo.net/' }]
          },
          { 
            id: 't-2-1-3', 
            title: 'Data Cleaning Pipelines', 
            points: ['Handling Missing Data', 'Outlier Detection', 'Data Imputation Strategies', 'Record Deduplication'],
            resources: [{ title: 'Kaggle: Data Cleaning Course', url: 'https://www.kaggle.com/learn/data-cleaning' }, { title: 'Data Cleaning with Python', url: 'https://realpython.com/python-data-cleaning-numpy-pandas/' }]
          },
        ],
      },
      {
        id: 'mod-2-2',
        title: 'Feature Engineering',
        subtopics: [
            { id: 't-2-2-1', title: 'Exploratory Data Analysis', points: ['Univariate Analysis', 'Correlation Matrices', 'Distribution Plots', 'Pair Plots'], resources: [{ title: 'Comprehensive EDA with Python', url: 'https://www.kaggle.com/code/pmarcelino/comprehensive-data-exploration-with-python' }] },
            { id: 't-2-2-2', title: 'Encoding & Scaling', points: ['One-Hot Encoding', 'Label Encoding', 'Min-Max Scaling', 'Z-Score Standardization'], resources: [{ title: 'Preprocessing Data (Scikit-Learn)', url: 'https://scikit-learn.org/stable/modules/preprocessing.html' }] },
            { id: 't-2-2-3', title: 'Dimensionality Reduction', points: ['Principal Component Analysis (PCA)', 't-SNE Visualization', 'Variance Retention', 'Feature Selection'], resources: [{ title: 'StatQuest: PCA Explained', url: 'https://www.youtube.com/watch?v=FgakZw6K1QQ' }] }
        ]
      }
    ]
  },
  {
    id: 'stage-3',
    title: 'STAGE 3 – CLASSICAL ML',
    modules: [
      {
        id: 'mod-3-1',
        title: 'Supervised Learning',
        subtopics: [
            { id: 't-3-1-1', title: 'Linear Regression Models', points: ['Simple Linear Regression', 'Multiple Regression', 'Cost Functions (MSE)', 'Gradient Descent'], resources: [{ title: 'Andrew Ng: Linear Regression', url: 'https://www.coursera.org/learn/machine-learning' }] },
            { id: 't-3-1-2', title: 'Classification Algorithms', points: ['Logistic Regression', 'SVM', 'KNN', 'Decision Boundaries'], resources: [{ title: 'Scikit-Learn Supervised Learning', url: 'https://scikit-learn.org/stable/supervised_learning.html' }] },
            { id: 't-3-1-3', title: 'Tree-Based Methods', points: ['Decision Trees', 'Random Forests', 'Gradient Boosting', 'XGBoost'], resources: [{ title: 'XGBoost Documentation', url: 'https://xgboost.readthedocs.io/en/stable/' }] }
        ]
      },
      {
        id: 'mod-3-2',
        title: 'Model Evaluation',
        subtopics: [
            { id: 't-3-2-1', title: 'Performance Metrics', points: ['Accuracy', 'Precision/Recall', 'ROC & AUC', 'F1-Score'], resources: [{ title: 'Classification Metrics Guide', url: 'https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ffa203121' }] },
            { id: 't-3-2-2', title: 'Validation Strategies', points: ['Train-Test Split', 'K-Fold Cross-Validation', 'Stratified Sampling', 'Bias-Variance Tradeoff'], resources: [{ title: 'Cross Validation (Sklearn)', url: 'https://scikit-learn.org/stable/modules/cross_validation.html' }] },
            { id: 't-3-2-3', title: 'Hyperparameter Tuning', points: ['Grid Search', 'Random Search', 'Bayesian Optimization', 'Optuna'], resources: [{ title: 'Hyperparameter Tuning with Optuna', url: 'https://optuna.org/' }] }
        ]
      }
    ]
  },
  {
    id: 'stage-4',
    title: 'STAGE 4 – DEEP LEARNING',
    modules: [
        {
            id: 'mod-4-1', title: 'Neural Networks',
            subtopics: [
                { id: 't-4-1-1', title: 'Perceptron Architecture', points: ['Neuron Anatomy', 'Weights & Biases', 'Forward Propagation', 'Step Functions'], resources: [{ title: 'Neural Networks and Deep Learning', url: 'http://neuralnetworksanddeeplearning.com/chap1.html' }] },
                { id: 't-4-1-2', title: 'Backpropagation', points: ['Computational Graphs', 'Chain Rule', 'Weight Updates', 'Error Minimization'], resources: [{ title: 'Andrej Karpathy: Micrograd', url: 'https://www.youtube.com/watch?v=VMj-3S1tku0' }] },
                { id: 't-4-1-3', title: 'Activation Functions', points: ['Sigmoid & Tanh', 'ReLU & Leaky ReLU', 'Softmax', 'Vanishing Gradient'], resources: [{ title: 'Activation Functions Explained', url: 'https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6' }] }
            ]
        },
        {
            id: 'mod-4-2', title: 'Advanced Architectures',
            subtopics: [
                { id: 't-4-2-1', title: 'CNNs', points: ['Convolutional Layers', 'Pooling', 'Padding & Stride', 'ResNet & VGG'], resources: [{ title: 'CS231n: CNNs', url: 'https://cs231n.github.io/' }] },
                { id: 't-4-2-2', title: 'RNNs', points: ['Recurrent Neurons', 'LSTM', 'GRU', 'Seq2Seq'], resources: [{ title: 'Understanding LSTMs', url: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/' }] },
                { id: 't-4-2-3', title: 'Optimizers', points: ['SGD', 'Momentum', 'RMSprop', 'Adam'], resources: [{ title: 'PyTorch Optimizers', url: 'https://pytorch.org/docs/stable/optim.html' }] }
            ]
        }
    ]
  },
  {
    id: 'stage-5',
    title: 'STAGE 5 – TRANSFORMERS & LLMs',
    modules: [
        {
            id: 'mod-5-1', title: 'Attention Mechanisms',
            subtopics: [
                { id: 't-5-1-1', title: 'Self-Attention', points: ['Query, Key, Value', 'Scaled Dot-Product', 'Multi-Head Attention', 'Masks'], resources: [{ title: 'The Illustrated Transformer', url: 'https://jalammar.github.io/illustrated-transformer/' }] },
                { id: 't-5-1-2', title: 'Transformer Architecture', points: ['Encoder-Decoder', 'Positional Encoding', 'Feed-Forward', 'Normalization'], resources: [{ title: 'Attention Is All You Need', url: 'https://arxiv.org/abs/1706.03762' }] },
                { id: 't-5-1-3', title: 'BERT & GPT', points: ['Bidirectional Encoders', 'Generative Pre-training', 'Masked LM', 'Next Token Prediction'], resources: [{ title: 'BERT Explained', url: 'https://jalammar.github.io/illustrated-bert/' }] }
            ]
        },
        {
            id: 'mod-5-2', title: 'GenAI Engineering',
            subtopics: [
                { id: 't-5-2-1', title: 'Prompt Engineering', points: ['Zero-Shot', 'Chain of Thought', 'Context Windows', 'System Instructions'], resources: [{ title: 'Prompt Engineering Guide', url: 'https://www.promptingguide.ai/' }] },
                { id: 't-5-2-2', title: 'RAG Systems', points: ['Vector Embeddings', 'Vector Databases', 'Semantic Search', 'Retrieval'], resources: [{ title: 'LangChain RAG Tutorial', url: 'https://python.langchain.com/docs/use_cases/question_answering/' }] },
                { id: 't-5-2-3', title: 'Fine-Tuning', points: ['Transfer Learning', 'PEFT', 'LoRA & QLoRA', 'Catastrophic Forgetting'], resources: [{ title: 'Hugging Face PEFT', url: 'https://huggingface.co/docs/peft/index' }] }
            ]
        }
    ]
  },
  {
    id: 'stage-6',
    title: 'STAGE 6 – SPECIALIZATIONS',
    modules: [
        {
            id: 'mod-6-1', title: 'Advanced Domains',
            subtopics: [
                { id: 't-6-1-1', title: 'Computer Vision', points: ['Object Detection (YOLO)', 'Segmentation', 'GANs', 'Stable Diffusion'], resources: [{ title: 'YOLOv8 Docs', url: 'https://docs.ultralytics.com/' }] },
                { id: 't-6-1-2', title: 'NLP', points: ['Tokenization', 'NER', 'Sentiment Analysis', 'Translation'], resources: [{ title: 'Hugging Face NLP Course', url: 'https://huggingface.co/learn/nlp-course' }] },
                { id: 't-6-1-3', title: 'Reinforcement Learning', points: ['Agents', 'MDP', 'Q-Learning', 'RLHF'], resources: [{ title: 'OpenAI Spinning Up', url: 'https://spinningup.openai.com/en/latest/' }] }
            ]
        }
    ]
  },
  {
    id: 'stage-7',
    title: 'STAGE 7 – PRODUCTION & MLOPS',
    modules: [
        {
            id: 'mod-7-1', title: 'Deployment',
            subtopics: [
                { id: 't-7-1-1', title: 'Model Serving', points: ['REST APIs (FastAPI)', 'gRPC', 'Serialization', 'Latency'], resources: [{ title: 'FastAPI Tutorial', url: 'https://fastapi.tiangolo.com/tutorial/' }] },
                { id: 't-7-1-2', title: 'Containerization', points: ['Docker', 'Kubernetes', 'Microservices', 'Dependency Mgmt'], resources: [{ title: 'Docker for Data Science', url: 'https://towardsdatascience.com/docker-for-data-science-9c0ce73e8263' }] },
                { id: 't-7-1-3', title: 'Cloud', points: ['Serverless', 'Managed Services', 'Auto-scaling', 'Load Balancing'], resources: [{ title: 'AWS Machine Learning', url: 'https://aws.amazon.com/machine-learning/' }] }
            ]
        },
        {
            id: 'mod-7-2', title: 'Operations',
            subtopics: [
                { id: 't-7-2-1', title: 'CI/CD for ML', points: ['Automated Testing', 'Versioning', 'GitHub Actions', 'DVC'], resources: [{ title: 'MLOps with GitHub Actions', url: 'https://github.blog/2020-06-17-machine-learning-with-github-actions/' }] },
                { id: 't-7-2-2', title: 'Monitoring', points: ['Data Drift', 'Concept Drift', 'System Latency', 'Error Rates'], resources: [{ title: 'Evidently AI', url: 'https://www.evidentlyai.com/' }] },
                { id: 't-7-2-3', title: 'Ethics & Safety', points: ['Bias', 'Fairness', 'Explainability (SHAP)', 'Adversarial Attacks'], resources: [{ title: 'Google AI Principles', url: 'https://ai.google/principles/' }] }
            ]
        }
    ]
  }
];

/**
 * STATE MANAGEMENT
 */
let state = {
    completedSubtopics: JSON.parse(localStorage.getItem('ai-roadmap-subtopics')) || [],
    activeModalNode: null,
    containerWidth: window.innerWidth,
    isMobile: window.innerWidth < 768
};

// Utils
function getCompletedModules() {
    const completed = [];
    CURRICULUM.forEach(stage => {
        stage.modules.forEach(module => {
            const allSubtopicsDone = module.subtopics.every(st => state.completedSubtopics.includes(st.id));
            if (allSubtopicsDone) completed.push(module.id);
        });
    });
    return completed;
}

function getActiveModule() {
    const completed = getCompletedModules();
    for (const stage of CURRICULUM) {
        for (const module of stage.modules) {
            if (!completed.includes(module.id)) return module.id;
        }
    }
    return null; // All done
}

function getNodeStatus(nodeId) {
    const completed = getCompletedModules();
    const active = getActiveModule();
    if (completed.includes(nodeId)) return 'COMPLETED';
    if (nodeId === active) return 'ACTIVE';
    return 'LOCKED';
}

function saveState() {
    localStorage.setItem('ai-roadmap-subtopics', JSON.stringify(state.completedSubtopics));
    render();
}

/**
 * RENDERERS
 */

function renderHeader() {
    const totalSubtopics = CURRICULUM.reduce((acc, stage) => acc + stage.modules.reduce((mAcc, mod) => mAcc + mod.subtopics.length, 0), 0);
    const progressPercent = Math.round((state.completedSubtopics.length / totalSubtopics) * 100);
    
    document.getElementById('progress-text').textContent = `${progressPercent}%`;
    document.getElementById('progress-bar').style.width = `${progressPercent}%`;

    const badge = document.getElementById('completion-badge');
    if (progressPercent === 100) badge.classList.remove('hidden');
    else badge.classList.add('hidden');
}

function renderRoadmap() {
    const container = document.getElementById('roadmap-container');
    container.innerHTML = '';
    
    // Layout Calculation
    const nodes = [];
    let currentY = 140;
    const nodeSpacing = 160;
    const defaultStageGap = 160;
    const largeStageGap = 400;
    let nodeIndex = 0;
    const amplitude = 30; // 30% width variance

    const completedIds = getCompletedModules();
    const activeId = getActiveModule();

    // 1. Calculate Positions
    CURRICULUM.forEach((stage, stageIndex) => {
        if (stageIndex > 0) {
            const gap = stageIndex === 1 ? largeStageGap : defaultStageGap;
            currentY += gap;
        }

        stage.modules.forEach(module => {
            let x = 50;
            if (!state.isMobile) {
                const direction = nodeIndex % 2 === 0 ? -1 : 1;
                x = 50 + (direction * amplitude);
            }

            nodes.push({
                ...module,
                stageTitle: stage.title,
                x, 
                y: currentY,
                status: getNodeStatus(module.id)
            });
            currentY += nodeSpacing;
            nodeIndex++;
        });
    });

    container.style.height = `${currentY + 200}px`;

    // 2. Generate SVG Path
    let pathD = '';
    if (nodes.length > 0) {
        pathD = `M ${nodes[0].x}% ${nodes[0].y}`;
        for (let i = 0; i < nodes.length - 1; i++) {
            const curr = nodes[i];
            const next = nodes[i + 1];
            if (state.isMobile) {
                pathD += ` L ${next.x}% ${next.y}`;
            } else {
                pathD += ` C ${curr.x}% ${(curr.y + next.y)/2}, ${next.x}% ${(curr.y + next.y)/2}, ${next.x}% ${next.y}`;
            }
        }
    }

    // 3. Generate Active Glow Path
    let activePathD = '';
    const activeNodeIndex = nodes.findIndex(n => n.id === activeId);
    let targetIndex = activeNodeIndex === -1 && completedIds.length > 0 ? nodes.length - 1 : activeNodeIndex;
    
    if (targetIndex !== -1 && nodes.length > 0) {
        activePathD = `M ${nodes[0].x}% ${nodes[0].y}`;
        for (let i = 0; i < targetIndex; i++) {
            if (i >= nodes.length - 1) break;
            const curr = nodes[i];
            const next = nodes[i + 1];
            if (state.isMobile) {
                activePathD += ` L ${next.x}% ${next.y}`;
            } else {
                activePathD += ` C ${curr.x}% ${(curr.y + next.y)/2}, ${next.x}% ${(curr.y + next.y)/2}, ${next.x}% ${next.y}`;
            }
        }
    }

    // 4. Build SVG Layer
    const svgHTML = `
        <svg class="absolute top-0 left-0 w-full h-full pointer-events-none z-0 overflow-visible">
            <defs>
                <filter id="neon-glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                    <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
                <linearGradient id="pathGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#334155" stopOpacity="0" />
                    <stop offset="10%" stopColor="#334155" stopOpacity="1" />
                    <stop offset="90%" stopColor="#334155" stopOpacity="1" />
                    <stop offset="100%" stopColor="#334155" stopOpacity="0" />
                </linearGradient>
            </defs>
            <path d="${pathD}" fill="none" stroke="url(#pathGradient)" stroke-width="${state.isMobile ? 4 : 8}" stroke-linecap="round" />
            <path d="${activePathD}" fill="none" stroke="#f97316" stroke-width="${state.isMobile ? 4 : 6}" stroke-linecap="round" filter="url(#neon-glow)" class="opacity-80 transition-all duration-1000" stroke-dasharray="10 5">
                <animate attributeName="stroke-dashoffset" from="100" to="0" dur="2s" repeatCount="indefinite" />
            </path>
        </svg>
    `;

    // 5. Build Stage Markers
    let markersHTML = '';
    let processedNodes = 0;
    CURRICULUM.forEach((stage, index) => {
        const firstNode = nodes[processedNodes];
        if (firstNode) {
            let prevGap = index === 1 ? largeStageGap : defaultStageGap;
            const offset = index === 0 ? 90 : (prevGap / 2);
            const top = firstNode.y - offset;
            
            markersHTML += `
                <div class="absolute left-0 w-full flex justify-center pointer-events-none z-0" style="top: ${top}px">
                    <div class="relative w-full flex justify-center items-center">
                        <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full max-w-4xl px-8 flex items-center justify-center">
                            <div class="h-px bg-gradient-to-r from-transparent via-slate-700 to-transparent flex-1 opacity-50"></div>
                            <div class="w-32"></div>
                            <div class="h-px bg-gradient-to-r from-transparent via-slate-700 to-transparent flex-1 opacity-50"></div>
                        </div>
                        <div class="relative px-6 py-2 rounded-full border border-slate-800/80 bg-slate-950/80 backdrop-blur text-slate-400 font-bold tracking-[0.2em] text-xs uppercase shadow-lg z-10">
                            ${stage.title}
                        </div>
                    </div>
                </div>
            `;
            processedNodes += stage.modules.length;
        }
    });

    // 6. Build Nodes
    const nodesHTML = nodes.map(node => {
        const isLocked = node.status === 'LOCKED';
        const isCompleted = node.status === 'COMPLETED';
        const isActive = node.status === 'ACTIVE';

        let btnClass = "relative flex items-center justify-center rounded-full transition-all duration-300 cursor-pointer z-10 border-4";
        let sizeClass = state.isMobile ? "w-20 h-20" : "w-24 h-24";
        let colorClass = "";
        let glowClass = "";
        let iconHTML = "";

        if (isLocked) {
            colorClass = "bg-slate-950 border-slate-800 text-slate-700 hover:border-slate-700";
            iconHTML = `<i data-lucide="lock" class="w-7 h-7"></i>`;
        } else if (isCompleted) {
            colorClass = "bg-slate-900 border-emerald-500 text-emerald-500 hover:scale-105 hover:bg-emerald-950";
            glowClass = "shadow-[0_0_15px_rgba(16,185,129,0.3)]";
            iconHTML = `<i data-lucide="check" class="w-9 h-9 stroke-[3px]"></i>`;
        } else if (isActive) {
            colorClass = "bg-slate-900 border-orange-500 text-orange-500 hover:scale-105 hover:bg-orange-950";
            glowClass = "shadow-[0_0_30px_rgba(249,115,22,0.5)] animate-glow";
            iconHTML = `<i data-lucide="zap" class="w-9 h-9 stroke-[3px] drop-shadow-[0_0_8px_rgba(249,115,22,0.8)]"></i>`;
        }

        const clickHandler = isLocked ? '' : `onclick="openModal('${node.id}')"`;
        const opacity = isLocked ? 'opacity-40 grayscale' : 'opacity-100';
        const textClass = isActive ? 'text-white' : (isCompleted ? 'text-emerald-400' : 'text-slate-300');

        return `
            <div class="absolute transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center group" style="left: ${node.x}%; top: ${node.y}px">
                <button ${clickHandler} class="${btnClass} ${sizeClass} ${colorClass} ${glowClass}">
                    ${iconHTML}
                </button>
                <div class="mt-5 text-center w-48 transition-all duration-300 ${opacity}">
                    <h4 class="text-base font-bold leading-tight mb-1 ${textClass}">${node.title}</h4>
                    <p class="text-[10px] font-bold tracking-widest text-slate-500 uppercase">${node.subtopics.length} Concepts</p>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = markersHTML + svgHTML + nodesHTML;
    lucide.createIcons();
}

function openModal(nodeId) {
    const stage = CURRICULUM.find(s => s.modules.some(m => m.id === nodeId));
    const node = stage.modules.find(m => m.id === nodeId);
    state.activeModalNode = node;
    renderModal();
    document.getElementById('modal-backdrop').classList.remove('hidden');
}

function closeModal() {
    state.activeModalNode = null;
    document.getElementById('modal-backdrop').classList.add('hidden');
}

function toggleSubtopic(id) {
    if (state.completedSubtopics.includes(id)) {
        state.completedSubtopics = state.completedSubtopics.filter(tid => tid !== id);
    } else {
        state.completedSubtopics.push(id);
    }
    saveState();
    // Re-render only modal contents if open, otherwise full render called by saveState
    if(state.activeModalNode) renderModal();
}

function renderModal() {
    if (!state.activeModalNode) return;
    const node = state.activeModalNode;
    const container = document.getElementById('modal-content');
    
    // Calculate progress
    const total = node.subtopics.length;
    const completedCount = node.subtopics.filter(st => state.completedSubtopics.includes(st.id)).length;
    const isModuleComplete = completedCount === total;
    const progressPercent = Math.round((completedCount / total) * 100);
    
    // Find Stage Title
    const stage = CURRICULUM.find(s => s.modules.some(m => m.id === node.id));

    const subtopicsHTML = node.subtopics.map(topic => {
        const isChecked = state.completedSubtopics.includes(topic.id);
        const cardClass = isChecked ? 'bg-emerald-950/20 border-emerald-500/30' : 'bg-slate-800/30 border-slate-700';
        const iconClass = isChecked ? 'text-emerald-500' : 'text-slate-600 group-hover:text-orange-500';
        const titleClass = isChecked ? 'text-emerald-200 line-through decoration-emerald-500/50' : 'text-slate-200 group-hover:text-white';
        const iconName = isChecked ? 'check-circle' : 'circle';
        
        // Resources
        let resourcesHTML = '';
        if (topic.resources && topic.resources.length > 0) {
            resourcesHTML = `
                <div class="mt-4 pt-3 border-t ${isChecked ? 'border-emerald-500/20' : 'border-slate-700/50'} ml-10">
                    <p class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Study Resources</p>
                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        ${topic.resources.map(res => `
                            <a href="${res.url}" target="_blank" class="flex items-center text-xs p-2 rounded bg-slate-900/50 hover:bg-slate-800 border border-slate-800 hover:border-slate-600 transition-all group/link">
                                <i data-lucide="external-link" class="mr-2 w-3 h-3 text-orange-500 group-hover/link:text-orange-400"></i>
                                <span class="text-slate-400 group-hover/link:text-slate-200 truncate">${res.title}</span>
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        return `
            <div class="group relative p-5 rounded-xl border-2 transition-all duration-200 ${cardClass}">
                <div class="flex items-start space-x-4 cursor-pointer" onclick="toggleSubtopic('${topic.id}')">
                    <div class="mt-1 flex-shrink-0 transition-colors ${iconClass}">
                        <i data-lucide="${iconName}" class="w-6 h-6"></i>
                    </div>
                    <div class="flex-1">
                        <h3 class="font-bold text-lg mb-2 transition-colors ${titleClass}">${topic.title}</h3>
                        <ul class="space-y-1 mb-4">
                            ${topic.points.map(p => `
                                <li class="flex items-start text-sm">
                                    <i data-lucide="chevron-right" class="mt-1 mr-2 w-3.5 h-3.5 flex-shrink-0 ${isChecked ? 'text-emerald-500/40' : 'text-orange-500/60'}"></i>
                                    <span class="${isChecked ? 'text-emerald-500/50' : 'text-slate-400'}">${p}</span>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                </div>
                ${resourcesHTML}
            </div>
        `;
    }).join('');

    container.innerHTML = `
        <div class="p-8 pb-4 border-b border-slate-800 bg-gradient-to-b from-slate-800/50 to-slate-900/50">
            <div class="flex items-start justify-between mb-4">
                <div>
                    <p class="text-orange-500 text-xs font-bold tracking-widest uppercase mb-2">${stage.title}</p>
                    <h2 class="text-3xl font-bold text-white mb-2">${node.title}</h2>
                </div>
                <button onclick="closeModal()" class="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-full transition-colors">
                    <i data-lucide="x" class="w-6 h-6"></i>
                </button>
            </div>
            <div class="flex items-center space-x-4">
                <div class="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div class="h-full transition-all duration-500 ${isModuleComplete ? 'bg-emerald-500' : 'bg-orange-500'}" style="width: ${progressPercent}%"></div>
                </div>
                <span class="text-sm font-mono font-bold ${isModuleComplete ? 'text-emerald-500' : 'text-orange-500'}">${completedCount}/${total}</span>
            </div>
        </div>
        <div class="p-8 overflow-y-auto space-y-4 max-h-[60vh] custom-scroll">
            <p class="text-slate-400 mb-6 text-sm">Study each concept below. Check the box when you are confident in your understanding to progress.</p>
            ${subtopicsHTML}
        </div>
        <div class="p-6 border-t border-slate-800 bg-slate-900 flex justify-between items-center">
             <div class="flex items-center text-slate-500 text-xs font-bold uppercase tracking-wider">
                <i data-lucide="book-open" class="mr-2 w-3.5 h-3.5"></i>
                Est. Time: ${total * 45} mins
             </div>
             <button onclick="closeModal()" class="px-6 py-3 rounded-xl font-bold transition-all ${isModuleComplete ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/20' : 'text-slate-400 hover:text-white hover:bg-slate-800'}">
                ${isModuleComplete ? 'Complete' : 'Keep Studying'}
             </button>
        </div>
    `;
    lucide.createIcons();
}

/**
 * INIT & LISTENERS
 */
function render() {
    renderHeader();
    renderRoadmap();
}

// Initial Render
render();

// Resize Handler
window.addEventListener('resize', () => {
    const newIsMobile = window.innerWidth < 768;
    if (newIsMobile !== state.isMobile) {
        state.isMobile = newIsMobile;
        renderRoadmap();
    }
});