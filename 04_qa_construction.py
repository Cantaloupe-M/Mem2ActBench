from __future__ import annotations
import argparse, json, logging, re, difflib, time, os, pickle, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import random
import threading
from datetime import datetime
from jsonschema import validate, ValidationError, Draft7Validator
import numpy as np
from rapidfuzz import fuzz
RAPIDFUZZ_AVAILABLE = True
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
CROSSENCODER_AVAILABLE = True
from rank_bm25 import BM25Okapi
import pandas as pd
import ast

BM25_AVAILABLE = True
GLINER_AVAILABLE = True
SEMANTIC_MODEL = None  # 延迟加载
SEMANTIC_AVAILABLE = True

# os.environ["OPENAI_API_KEY"] = "sk-fLtrd6nJYPnyfnzRCf7aD4DdE7C04aCbA85bCb4cC2C1329a"
# os.environ["OPENAI_BASE_URL"] = "https://api.apiyi.com/v1"
os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-omutivisxktqoglinfimxbckcxacgaqayusvamjquihiujov"

# 固定随机种子以保证可复现性
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)

class LLMClient:
    """统一的LLM调用类，支持指数退避重试"""
    def __init__(self, base_url: str = None, api_key: str = None, max_retries: int = 10, 
                 initial_delay: float = 1.0, max_delay: float = 60.0):
        self.client = OpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
    
    def chat_completion(self, model: str, messages: List[Dict], 
                       temperature: float = 0, response_format: Optional[Dict] = None,
                       **kwargs) -> Any:
        """带指数退避的聊天补全调用"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                    **kwargs
                )
                return response
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # 计算退避延迟：指数增长 + 随机抖动
                    delay = min(self.initial_delay * (2 ** attempt), self.max_delay)
                    jitter = random.uniform(0, delay * 0.1)  # 10%的随机抖动
                    total_delay = delay + jitter
                    
                    LOGGER.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    LOGGER.info(f"等待 {total_delay:.2f}秒后重试...")
                    time.sleep(total_delay)
                else:
                    LOGGER.error(f"LLM调用失败，已达最大重试次数: {e}")
        
        raise last_exception

def safe_json_loads(payload: str | bytes | None, default: Any) -> Any:
    if not payload: return default
    if isinstance(payload, bytes): payload = payload.decode("utf-8", errors="ignore")
    payload = re.sub(r"^```json\s*", "", payload, flags=re.MULTILINE)
    payload = re.sub(r"^```\s*", "", payload, flags=re.MULTILINE)
    payload = re.sub(r"```$", "", payload, flags=re.MULTILINE).strip()
    try: return json.loads(payload)
    except: return default

def split_ids(raw_ids: str | None) -> List[str]:
    if not raw_ids: return []
    parts = re.split(r"[,，]", str(raw_ids))
    return [p.strip() for p in parts if p.strip()]

class SchemaValidator:
    """JSON Schema 校验器 - 确保参数严格符合工具定义"""
    
    @staticmethod
    def validate_tool_call(tool: Dict, arguments: Dict) -> Tuple[bool, str]:
        """校验工具调用的参数是否符合schema
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # 提取参数schema
            parameters = tool.get("parameters", {})
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            
            # 1. 检查必填参数是否存在
            for req_param in required:
                if req_param not in arguments:
                    return False, f"缺少必填参数: {req_param}"
            
            # 2. 检查参数类型和格式（包括嵌套结构）
            for param_name, param_value in arguments.items():
                if param_name not in properties:
                    return False, f"未定义的参数: {param_name}"
                
                param_schema = properties[param_name]
                
                # 递归验证参数值
                is_valid, error = SchemaValidator._validate_value(
                    param_value, param_schema, param_name
                )
                if not is_valid:
                    return False, error
            
            return True, ""
        
        except Exception as e:
            return False, f"Schema校验异常: {str(e)}"
    
    @staticmethod
    def _validate_value(value: any, schema: Dict, path: str) -> Tuple[bool, str]:
        """递归验证值是否符合schema定义
        
        Args:
            value: 要验证的值
            schema: 对应的schema定义
            path: 当前路径（用于错误信息）
        
        Returns:
            (is_valid, error_message)
        """
        param_type = schema.get("type")
        
        # 基础类型检查
        if param_type == "string":
            if not isinstance(value, str):
                return False, f"参数 {path} 应为字符串类型，实际为 {type(value).__name__}"
        
        elif param_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"参数 {path} 应为整数类型，实际为 {type(value).__name__}"
        
        elif param_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"参数 {path} 应为数值类型，实际为 {type(value).__name__}"
        
        elif param_type == "boolean":
            if not isinstance(value, bool):
                return False, f"参数 {path} 应为布尔类型，实际为 {type(value).__name__}"
        
        elif param_type == "array":
            if not isinstance(value, list):
                return False, f"参数 {path} 应为数组类型，实际为 {type(value).__name__}"
            
            # 验证数组元素
            items_schema = schema.get("items", {})
            if items_schema:
                for idx, item in enumerate(value):
                    is_valid, error = SchemaValidator._validate_value(
                        item, items_schema, f"{path}[{idx}]"
                    )
                    if not is_valid:
                        return False, error
        
        elif param_type == "object":
            if not isinstance(value, dict):
                return False, f"参数 {path} 应为对象类型，实际为 {type(value).__name__}"
            
            # 验证对象的properties和required
            obj_properties = schema.get("properties", {})
            obj_required = schema.get("required", [])
            
            # 检查必填字段
            for req_field in obj_required:
                if req_field not in value:
                    return False, f"参数 {path} 缺少必填字段: {req_field}"
            
            # 递归验证每个字段
            for field_name, field_value in value.items():
                if field_name not in obj_properties:
                    return False, f"参数 {path} 包含未定义的字段: {field_name}"
                
                field_schema = obj_properties[field_name]
                is_valid, error = SchemaValidator._validate_value(
                    field_value, field_schema, f"{path}.{field_name}"
                )
                if not is_valid:
                    return False, error
        
        # enum检查
        if "enum" in schema:
            if value not in schema["enum"]:
                return False, f"参数 {path} 的值 '{value}' 不在允许的枚举值 {schema['enum']} 中"
        
        return True, ""
    

class MemoryGroundingValidator:
    """记忆锚定验证器 - 支持「硬+软」两级匹配
    
    匹配策略：
    - strict 模式：只使用精确匹配（100% 可靠的 gold 样本）
    - hybrid 模式：精确匹配 + 软匹配（扩大数据集，标记匹配类型）
    
    软匹配方法（按优先级）：
    1. 日期格式转换
    2. Token overlap (≥80%)
    3. RapidFuzz模糊匹配 (≥85%)
    4. 语义相似度 (≥0.75, 可选)
    """
    
    # 语义模型缓存（延迟加载）
    _semantic_model = None
    _semantic_lock = threading.Lock()
    
    # 常用英文停用词（用于软匹配的归一化）
    STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
                 "may", "might", "can", "of", "in", "on", "at", "to", "for", "with", "by", "from",
                 "and", "or", "but", "not", "as", "if", "that", "this", "it"}
    
    @staticmethod
    def validate_grounding(arguments: Dict, memory_chain: List[Dict], 
                          required_params: List[str], llm_grounded_params: List[str] = None,
                          mode: str = "strict", use_semantic: bool = False) -> Tuple[bool, List[str], Dict[str, str], Dict[str, str]]:
        """验证参数是否锚定在记忆中
        
        Args:
            arguments: 工具调用的参数字典
            memory_chain: 记忆演化链
            required_params: 必填参数列表
            llm_grounded_params: LLM声明的来自记忆的参数列表
            mode: 匹配模式 - "strict" (仅精确匹配) 或 "hybrid" (精确+软匹配)
            use_semantic: 是否启用语义相似度匹配（仅在hybrid模式下生效）
        
        Returns:
            (is_valid, grounded_params, param_to_memory_mapping, param_modes)
            - is_valid: 是否通过验证
            - grounded_params: 被锚定的参数名列表
            - param_to_memory_mapping: 参数到记忆fact_id的映射
            - param_modes: 参数到匹配模式的映射 {param_name: "exact" | "soft"}
        """
        grounded_params = []
        param_to_memory = {}
        param_modes = {} 
        
        # 提取所有记忆文本和fact
        memory_facts = []
        for item in memory_chain:
            fact = item.get("fact", "")
            if fact:
                memory_facts.append((fact, item.get("id", "")))
        
        params_to_validate = llm_grounded_params if llm_grounded_params else list(arguments.keys())
        
        for param_name in params_to_validate:
            if param_name not in arguments:
                continue
                
            param_value = arguments[param_name]
            param_str = str(param_value).strip()
            if not param_str:
                continue
            
            exact_hit = None
            soft_hit = None
            
            for fact, fact_id in memory_facts:
                # 优先尝试精确匹配
                if MemoryGroundingValidator._exact_match(param_str, fact):
                    exact_hit = (fact_id, "exact")
                    break
                
                # 如果是hybrid模式且未精确匹配，尝试软匹配
                if mode == "hybrid" and soft_hit is None:
                    if MemoryGroundingValidator._soft_match(param_str, fact, use_semantic=use_semantic):
                        soft_hit = (fact_id, "soft")
            
            # 选择匹配结果（精确匹配优先）
            hit = exact_hit or soft_hit
            if hit:
                fact_id, match_mode = hit
                grounded_params.append(param_name)
                param_to_memory[param_name] = fact_id
                param_modes[param_name] = match_mode
            else:
                # 声明的grounded参数未找到锚定，记录日志
                LOGGER.debug(f"声明的grounded参数 '{param_name}' 值 '{param_str}' 未在记忆中找到")

        if llm_grounded_params:
            # 新逻辑：LLM声明的每个grounded参数都必须在记忆中找到
            is_valid = len(grounded_params) == len(llm_grounded_params)
        else:
            # 如果没有声明grounded_params，则默认为无效
            is_valid = False

        return is_valid, grounded_params, param_to_memory, param_modes
    
    @staticmethod
    def _exact_match(value: str, text: str) -> bool:
        """精确匹配（硬匹配）：支持整词匹配和数值匹配
        
        策略：
        1. 直接包含检查（大小写敏感）
        2. 整词边界匹配（大小写不敏感）
        3. 数值精确匹配
        """
        # 直接包含检查（大小写敏感）
        if value in text:
            return True
        
        # 大小写不敏感的整词匹配
        pattern = r'\b' + re.escape(value) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return True
        
        # 数值匹配（如果是数字）
        try:
            val_num = float(value)
            # 在文本中查找相同数值
            if str(int(val_num)) in text or str(val_num) in text:
                return True
        except ValueError:
            pass
        
        return False
    
    @staticmethod
    def _soft_match(value: str, text: str, use_semantic: bool = False) -> bool:
        """软匹配：使用多种NLP方法处理格式变化、轻微改写
        
        策略（高阈值，宁可少收不要错收）：
        1. 日期格式转换（如 "2023-08-12" vs "Aug 12, 2023"）
        2. Token overlap >= 80%
        3. RapidFuzz模糊匹配 >= 85% (比difflib快10-100倍)
        4. 语义相似度 >= 0.75 (可选，需要use_semantic=True)
        
        Args:
            value: 待匹配的参数值
            text: 记忆文本
            use_semantic: 是否启用语义相似度匹配（较慢但更准确）
        
        注意：所有检查都有高阈值，避免误匹配
        """
        v = value.strip()
        t = text.strip()
        
        # 过滤掉过短的值（容易误报）
        if len(v) < 3:
            return False
        
        # === 策略 1: 日期格式匹配 ===
        # 检测常见日期格式：YYYY-MM-DD, YYYY/MM/DD
        if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", v):
            try:
                from datetime import datetime as dt
                # 解析目标日期
                target_date = dt.strptime(v.replace("/", "-"), "%Y-%m-%d").date()
                
                # 查找文本中的其他日期格式
                # 支持：Aug 12, 2023 | 12/08/2023 | 2023.08.12 等
                date_patterns = [
                    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
                    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                    r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b",
                ]
                
                for pattern in date_patterns:
                    for match in re.finditer(pattern, t, re.IGNORECASE):
                        candidate = match.group()
                        try:
                            # 尝试多种解析格式
                            for fmt in ["%b %d, %Y", "%b %d %Y", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]:
                                try:
                                    cand_date = dt.strptime(candidate, fmt).date()
                                    if cand_date == target_date:
                                        return True
                                except:
                                    continue
                        except:
                            continue
            except:
                pass
        
        # === 策略 2: Token Overlap（高门槛 80%）===
        def normalize_tokens(s: str) -> Set[str]:
            """归一化：小写、去标点、去停用词"""
            s_lower = s.lower()
            # 去除标点，保留字母数字
            s_clean = re.sub(r"[^\w\s]", " ", s_lower)
            tokens = s_clean.split()
            # 去除停用词
            filtered = [t for t in tokens if t and t not in MemoryGroundingValidator.STOPWORDS]
            return set(filtered)
        
        v_tokens = normalize_tokens(v)
        t_tokens = normalize_tokens(t)
        
        if v_tokens:  # 避免除零
            overlap = len(v_tokens & t_tokens) / len(v_tokens)
            if overlap >= 0.8:  # 80% 的 token 被命中
                LOGGER.debug(f"软匹配成功 (token overlap={overlap:.2f}): '{v}' in '{t[:100]}...'")
                return True
        
        # === 策略 3: RapidFuzz模糊匹配（高门槛 85%）===
        if RAPIDFUZZ_AVAILABLE:
            # RapidFuzz比difflib快10-100倍
            ratio = fuzz.partial_ratio(v.lower(), t.lower()) / 100.0
            if ratio >= 0.85:
                LOGGER.debug(f"软匹配成功 (rapidfuzz={ratio:.2f}): '{v}' ~ '{t[:100]}...'")
                return True
        else:
            # 降级到difflib（仅对短字符串）
            if len(v) <= 20:
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, v.lower(), t.lower()).ratio()
                if ratio >= 0.9:
                    LOGGER.debug(f"软匹配成功 (difflib={ratio:.2f}): '{v}' ~ '{t[:100]}...'")
                    return True
        
        # === 策略 4: 语义相似度匹配（可选，高质量但较慢）===
        if use_semantic and SEMANTIC_AVAILABLE:
            try:
                # 延迟加载模型（避免启动开销）
                if MemoryGroundingValidator._semantic_model is None:
                    with MemoryGroundingValidator._semantic_lock:
                        if MemoryGroundingValidator._semantic_model is None:
                            LOGGER.info("加载语义相似度模型 (all-MiniLM-L6-v2)...")
                            MemoryGroundingValidator._semantic_model = SentenceTransformer(
                                'sentence-transformers/all-MiniLM-L6-v2'
                            )
                
                model = MemoryGroundingValidator._semantic_model
                
                # 计算语义相似度
                emb_v = model.encode(v, convert_to_tensor=True)
                emb_t = model.encode(t[:512], convert_to_tensor=True)  # 限制长度
                similarity = util.cos_sim(emb_v, emb_t).item()
                
                if similarity >= 0.75:  # 高阈值
                    LOGGER.debug(f"软匹配成功 (semantic={similarity:.2f}): '{v}' ~ '{t[:100]}...'")
                    return True
            except Exception as e:
                LOGGER.warning(f"语义匹配失败: {e}")
        
        return False

class ToolPoolManager:
    """工具池管理器"""
    # 不适合QA构造的工具关键词模式
    EXCLUDED_TOOL_PATTERNS = [
        "login", "authenticate", "signin", "signup", "sign-in", "sign-up",
        "register", "password", "token", "logout", "auth"
    ]
    
    def __init__(self, tool_pool_path: Path, enable_advanced_retrieval: bool = True):
        self.tool_pool_path = Path(tool_pool_path)
        with open(tool_pool_path, "r", encoding="utf-8") as f:
            all_tools = json.load(f)
        
        # 过滤掉不适合的工具
        self.tools = [tool for tool in all_tools if self._is_tool_suitable(tool)]
        excluded_count = len(all_tools) - len(self.tools)
        
        # 初始化检索相关属性
        self.tool_embeddings = None  # Dense向量
        self.semantic_model = None   # Dense编码器
        self.bm25_index = None       # BM25索引
        self.tool_texts = []         # 工具文本缓存
        self.enable_advanced = enable_advanced_retrieval
        
        self._init_semantic_search()
        
        LOGGER.info(f"加载了 {len(self.tools)} 个工具 (排除了 {excluded_count} 个不适合的工具)")
        LOGGER.info(f"高级检索模式: {'启用' if self.enable_advanced else '禁用'}")
    
    def _is_tool_suitable(self, tool: Dict) -> bool:
        """判断工具是否适合用于QA构造"""
        if not isinstance(tool, dict):
            return False
        
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        # 检查是否包含排除的关键词
        for pattern in self.EXCLUDED_TOOL_PATTERNS:
            if pattern in tool_name or pattern in tool_desc:
                LOGGER.debug(f"排除工具: {tool.get('name')} (匹配模式: {pattern})")
                return False
        
        return True
    
    def search_tools_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[Dict]:
        """根据关键词搜索相关工具"""
        scored_tools = []
        for tool in self.tools:
            if isinstance(tool, dict):
                name = tool.get("name", "").lower()
                desc = tool.get("description", "").lower()
                score = sum(1 for kw in keywords if kw.lower() in name or kw.lower() in desc)
                if score > 0:
                    scored_tools.append((tool, score))
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in scored_tools[:top_k]]
    
    def _init_semantic_search(self):
        """初始化混合检索模型（Dense + Sparse）"""
        if not SEMANTIC_AVAILABLE:
            LOGGER.warning("sentence-transformers未安装，将使用基础规则匹配")
            return
            
        try:
            LOGGER.info("初始化语义检索模型...")
            self.semantic_model = SentenceTransformer('BAAI/bge-m3')
            
            cache_dir = Path(".cache")
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / "embeddings_cache_tool_pool.pkl"
            
            loaded_from_cache = False
            if cache_file.exists():
                try:
                    LOGGER.info(f"发现嵌入缓存: {cache_file}")
                    with open(cache_file, "rb") as f:
                        cache_data = pickle.load(f)
                        self.tool_texts = cache_data["tool_texts"]
                        self.tool_embeddings = cache_data["tool_embeddings"]
                        loaded_from_cache = True
                        LOGGER.info("✓ 缓存加载成功")
                except Exception as e:
                    LOGGER.warning(f"缓存加载失败: {e}")
            
            if not loaded_from_cache:
                LOGGER.info("计算工具嵌入向量...")
                # 构建工具文本（包含参数信息）
                self.tool_texts = []
                for tool in self.tools:
                    name = tool.get('name', '')
                    desc = tool.get('description', '')
                    category = tool.get('category_name', '')
                    
                    # 提取参数信息增强语义
                    params = tool.get('parameters', {}).get('properties', {})
                    param_texts = []
                    for p_name, p_info in params.items():
                        p_desc = p_info.get('description', '')
                        if p_desc:
                            param_texts.append(f"{p_name} ({p_desc})")
                        else:
                            param_texts.append(p_name)
                    param_str = ", ".join(param_texts) if param_texts else ""
                    
                    # 构建结构化文本
                    tool_text = f"Tool: {name}\nDescription: {desc}\nParameters: {param_str}\nCategory: {category}"
                    self.tool_texts.append(tool_text)
                
                # 生成 Dense embeddings
                self.tool_embeddings = self.semantic_model.encode(
                    self.tool_texts, 
                    batch_size=32,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=True
                )
                
                # 保存缓存
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump({
                            "tool_texts": self.tool_texts,
                            "tool_embeddings": self.tool_embeddings
                        }, f)
                    LOGGER.info(f"嵌入向量已保存至缓存: {cache_file}")
                except Exception as e:
                    LOGGER.warning(f"缓存保存失败: {e}")

            LOGGER.info(f"✓ Dense索引: {len(self.tools)} 个工具")
            
            # 构建 BM25 稀疏索引
            if BM25_AVAILABLE and self.enable_advanced:
                tokenized_corpus = [text.lower().split() for text in self.tool_texts]
                self.bm25_index = BM25Okapi(tokenized_corpus)
                LOGGER.info(f"✓ BM25索引: 已构建")
            elif not BM25_AVAILABLE:
                LOGGER.warning("rank-bm25未安装，稀疏检索已禁用（pip install rank-bm25）")
            
        except Exception as e:
            LOGGER.error(f"检索系统初始化失败: {e}，回退到基础规则匹配")
            self.semantic_model = None
            self.tool_embeddings = None
            self.bm25_index = None
    
    
    def hybrid_tool_selection(self, memory_chain: List[Dict], 
                             top_k: int = 5) -> List[Dict]:
        """混合语义检索工具选择（简化版）""" 
        memory_text = " ".join([item.get("fact", "") for item in memory_chain])
        
        # 混合检索（Dense + Sparse）
        candidates = self._hybrid_retrieval(memory_text, top_k=top_k * 3)  # 召回3倍候选
        
        if not candidates:
            LOGGER.warning("混合检索无结果，回退到关键词召回")
            return self._keyword_recall(memory_text, top_k=top_k)
        
        # 返回Top-K
        selected_tools = candidates[:top_k]
        return selected_tools
    
    def _hybrid_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        """Stage 1: 混合检索（Dense + Sparse融合）
        
        使用Reciprocal Rank Fusion (RRF) 融合算法
        参考：Cormack et al. SIGIR 2009
        """
        dense_scores = np.zeros(len(self.tools))
        sparse_scores = np.zeros(len(self.tools))
        
        if self.semantic_model and self.tool_embeddings is not None:
            query_emb = self.semantic_model.encode(
                query, normalize_embeddings=True, show_progress_bar=False
            )
            dense_scores = np.dot(self.tool_embeddings.cpu().numpy(), query_emb)
        
        if self.bm25_index:
            query_tokens = query.lower().split()
            sparse_scores = np.array(self.bm25_index.get_scores(query_tokens))
        
        hybrid_scores = self._reciprocal_rank_fusion(dense_scores, sparse_scores, k=60)
        
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        candidates = [self.tools[i] for i in top_indices]
        
        return candidates
    
    def _reciprocal_rank_fusion(self, scores1: np.ndarray, scores2: np.ndarray, k: int = 60) -> np.ndarray:
        """Reciprocal Rank Fusion融合算法
        
        RRF(item) = sum_i [1 / (k + rank_i(item))]
        
        优势：
        - 不需要归一化分数（不同检索器分数尺度不同）
        - 对排名靠前的结果敏感
        - 鲁棒性强（SIGIR 2009验证）
        """
        # 转换为排名（rank从0开始）
        rank1 = np.argsort(np.argsort(-scores1))  # 降序排名
        rank2 = np.argsort(np.argsort(-scores2))
        
        # RRF公式
        rrf_scores = 1.0 / (k + rank1) + 1.0 / (k + rank2)
        return rrf_scores
    
    
    def _semantic_recall(self, memory_text: str, top_k: int = 20) -> List[Dict]:
        """使用语义相似度召回相关工具"""
        try:
            memory_embedding = self.semantic_model.encode(
                memory_text, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            similarities = util.cos_sim(memory_embedding, self.tool_embeddings)[0]
            top_indices = similarities.topk(min(top_k, len(self.tools))).indices.tolist()
            candidates = [self.tools[idx] for idx in top_indices]
            
            # 可选：叠加主题过滤（进一步提升精准度）
            # topic_filtered = self._topic_filter(memory_text, candidates)
            # return topic_filtered if topic_filtered else candidates
            
            return candidates
            
        except Exception as e:
            LOGGER.error(f"语义召回失败: {e}")
            return []
    
    def _topic_filter(self, memory_text: str, candidates: List[Dict]) -> List[Dict]:
        """主题关键词过滤：根据记忆的话题领域筛选工具
        
        目的：排除语义相似但领域不符的工具
        例如：记忆聊"订餐"，排除"订机票"工具（虽然都是预订，但领域不同）
        
        策略：
        - 识别记忆中的领域关键词（food, flight, hotel, shopping等）
        - 只保留与领域匹配的工具
        - 如果没有明确领域，返回所有候选
        """
        # 定义领域关键词映射（可扩展）
        TOPIC_CATEGORIES = {
            "food": ["food", "restaurant", "meal", "dinner", "lunch", "breakfast", 
                     "hungry", "eat", "drink", "menu", "order", "cuisine"],
            "travel": ["flight", "hotel", "booking", "travel", "trip", "airport", 
                       "reservation", "check-in", "vacation", "destination"],
            "shopping": ["shop", "buy", "purchase", "cart", "product", "price", 
                         "discount", "order", "delivery", "payment"],
            "finance": ["payment", "transaction", "account", "balance", "transfer", 
                        "bank", "money", "credit", "debit", "invoice"],
            "communication": ["message", "email", "call", "contact", "notification", 
                             "send", "receive", "chat", "reply"],
        }
        
        memory_lower = memory_text.lower()
        
        # Step 1: 识别记忆中的领域
        detected_topics = set()
        for topic, keywords in TOPIC_CATEGORIES.items():
            if any(kw in memory_lower for kw in keywords):
                detected_topics.add(topic)
        
        # Step 2: 如果没有识别到明确领域，返回所有候选（避免误杀）
        if not detected_topics:
            LOGGER.debug("未检测到明确主题，跳过主题过滤")
            return candidates
        
        LOGGER.debug(f"检测到主题: {detected_topics}")
        
        # Step 3: 根据领域过滤工具
        filtered = []
        for tool in candidates:
            tool_name = tool.get("name", "").lower()
            tool_desc = tool.get("description", "").lower()
            tool_text = f"{tool_name} {tool_desc}"
            
            # 检查工具是否属于检测到的任一领域
            is_relevant = False
            for topic in detected_topics:
                topic_keywords = TOPIC_CATEGORIES.get(topic, [])
                if any(kw in tool_text for kw in topic_keywords):
                    is_relevant = True
                    break
            
            if is_relevant:
                filtered.append(tool)
            else:
                LOGGER.debug(f"主题过滤剔除: {tool.get('name')} (不属于 {detected_topics})")
        
        # Step 4: 如果过滤太严格导致结果过少，返回原候选
        if len(filtered) < 5:
            LOGGER.warning(f"主题过滤后仅剩 {len(filtered)} 个工具，回退到原候选")
            return candidates
        
        return filtered
    
    def _keyword_recall(self, memory_text: str, top_k: int = 20) -> List[Dict]:
        """关键词匹配召回（语义检索的fallback）"""
        memory_keywords = set()
        
        # 提取记忆中的关键词
        words = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b|\b[a-z]{4,}\b', memory_text)
        memory_keywords.update(words)
        
        scored_tools = []
        for tool in self.tools:
            score = self._compute_keyword_score(tool, memory_keywords)
            if score > 0:
                scored_tools.append((tool, score))
        
        # 按分数排序，取top_k
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in scored_tools[:top_k]]
    
    def _compute_keyword_score(self, tool: Dict, keywords: Set[str]) -> float:
        """计算关键词匹配分数（简化版）"""
        score = 0.0
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        for kw in keywords:
            kw_lower = str(kw).lower()
            if len(kw_lower) < 3:  # 跳过过短的词
                continue
            if kw_lower in tool_name:
                score += 3.0  # 名称匹配权重高
            elif kw_lower in tool_desc:
                score += 1.0  # 描述匹配权重低
        
        return score

    def _compute_relevance_score(self, tool: Dict, keywords: Set[str], 
                                memory_chain: List[Dict]) -> float:
        """计算工具相关性分数"""
        score = 0.0
        
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        # 1. 关键词匹配（名称和描述）
        for kw in keywords:
            kw_lower = str(kw).lower()
            if len(kw_lower) < 3:  # 跳过过短的词
                continue
            if kw_lower in tool_name:
                score += 3.0  # 名称匹配权重高
            elif kw_lower in tool_desc:
                score += 1.0  # 描述匹配权重低
        
        # 2. 检查至少一个必填参数能在记忆中找到候选值
        required_params = tool.get("parameters", {}).get("required", [])
        if required_params:
            memory_text = " ".join([item.get("fact", "") for item in memory_chain]).lower()
            
            for param in required_params:
                # 简单的启发式：参数名是否与记忆中的实体相关
                param_lower = param.lower()
                # 检查记忆中是否有相关信息
                for kw in keywords:
                    kw_lower = str(kw).lower()
                    if len(kw_lower) >= 3:
                        # 参数名与关键词相似
                        if param_lower in kw_lower or kw_lower in param_lower:
                            score += 2.0
                            break
        
        return score

class MemoryToToolMapper:
    """记忆到工具的映射器"""
    def __init__(self, model: str, tool_pool: ToolPoolManager):
        self.model = model
        self.tool_pool = tool_pool
        self.llm_client = LLMClient()
    
    def analyze_and_select_tool(self, memory_chain: List[Dict]) -> Optional[Dict]:
        """分析记忆并选择合适的工具
        
        策略（精简版）：
        1. 规则预选：使用正则表达式从记忆中提取关键词，将候选工具缩小到 K=5 个
        2. LLM离散选择：在预选集中选择一个 index（这是唯一的LLM调用）
        """
        memory_text = json.dumps(memory_chain, ensure_ascii=False, indent=2)
        
        # Step 1: 混合预选 - 三阶段高级检索
        candidate_tools = self.tool_pool.hybrid_tool_selection(
            memory_chain=memory_chain,
            top_k=10 
        )
        
        if not candidate_tools:
            LOGGER.debug("规则预选未找到合适工具")
            return None
        
        # Step 2: LLM离散选择 - 在有限候选中选一个
        tool_options = []
        for idx, tool in enumerate(candidate_tools):
            tool_options.append({
                "index": idx,
                "name": tool.get("name"),
                "description": tool.get("description", "")
            })
        
        select_prompt = f"""You are given {len(tool_options)} pre-selected tools. Choose the MOST suitable one for the user's memory context.

            User Memory:
            {memory_text}

            Pre-selected Tool Options:
            {json.dumps(tool_options, indent=2, ensure_ascii=False)}

            CRITICAL REQUIREMENTS:
            1. You MUST select exactly ONE tool by its index (0 to {len(tool_options)-1})
            2. Your selection is CONSTRAINED to these {len(tool_options)} tools only
            3. Do NOT invent new tools or tool names
            4. ENTITY TYPE MATCHING (IMPORTANT):
               - If memory mentions "user" → select user-related tools
               - If memory mentions "service provider" / "provider" → select provider-related tools  
               - If memory mentions "product" / "item" / "book" → select item-related tools
               - If memory mentions "order" / "booking" / "appointment" → select order/booking-related tools
               - PAY ATTENTION to the entity type and choose tools that operate on the SAME entity type

            Output JSON:
            {{
                "selected_tool_index": 0,  // Must be an integer from 0 to {len(tool_options)-1}
                "reasoning": "why this tool is most suitable (mention entity type matching)"
            }}
            """
        
        response = self.llm_client.chat_completion(
            model=self.model, temperature=0,
            messages=[{"role": "user", "content": select_prompt}],
            response_format={"type": "json_object"}, timeout=30
        )
        selection = safe_json_loads(response.choices[0].message.content, {})
        selected_index = selection.get("selected_tool_index")
        
        # 验证选择的index是否合法
        if not isinstance(selected_index, int) or selected_index < 0 or selected_index >= len(candidate_tools):
            LOGGER.warning(f"选择的工具索引不合法: {selected_index}")
            return None
        
        selected_tool = candidate_tools[selected_index]
        LOGGER.debug(f"选择工具: {selected_tool.get('name')} (索引 {selected_index}/{len(candidate_tools)})")
        
        return {
            "tool": selected_tool,
            "tool_selection_metadata": {
                "total_candidates": len(candidate_tools),
                "selected_index": selected_index,
                "selection_reasoning": selection.get("reasoning", "")
            }
        }

class ToolCallConstructor:
    """基于工具schema和记忆填充参数 - 强约束 + 严格验证"""
    def __init__(self, model: str, grounding_mode: str = "strict"):
        self.model = model
        self.llm_client = LLMClient()
        self.schema_validator = SchemaValidator()
        self.grounding_validator = MemoryGroundingValidator()
        self.grounding_mode = grounding_mode  # "strict" 或 "hybrid"
    
    def construct_call(self, tool: Dict, memory_chain: List[Dict]) -> Optional[Dict]:
        """构造工具调用，增强约束和验证
        
        策略：
        1. LLM从记忆中提取参数值
        2. Schema校验：类型、enum、必填字段
        3. 记忆锚定验证：
           - Explicit params: 严格/软匹配验证
           - Inferred params: LLM语义推理验证
        4. 返回验证通过的工具调用 + 附带锚定信息
        """
        # memory_text = "\n".join([item.get("fact", "") for item in memory_chain if item.get("fact")])
        tool_params_schema = tool.get("parameters", {})  # 完整的 parameters schema（包含嵌套结构）
        required = tool.get("parameters", {}).get("required", [])  # 提取必填参数列表
        
        # 动态生成输出示例，基于实际的 tool schema
        properties = tool_params_schema.get("properties", {})
        example_arguments = {}
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            if param_type == "array":
                example_arguments[param_name] = ["..."]
            elif param_type == "object" or param_type == "dict":
                example_arguments[param_name] = {"key": "..."}
            elif param_type in ["integer", "number"]:
                example_arguments[param_name] = "..."
            elif param_type == "boolean":
                example_arguments[param_name] = "true/false"
            else:  # string
                example_arguments[param_name] = "..."
        
        example_output = {
            "name": tool.get('name'),
            "arguments": example_arguments,
            "grounding_info": {
                "param_name": {
                    "source_text": "exact text from memory or reasoning basis or default value definition",
                    "type": "explicit/inferred/default"
                }
            },
            "relevant_source_ids": ["source_id_1", "source_id_2","..."]
        }
        
        prompt = f"""Extract parameter values from user memory to construct a tool call.

            Tool Name: {tool.get('name')}
            Tool Description: {tool.get('description', '')}

            Complete Parameter Schema:
            {json.dumps(tool_params_schema, indent=2, ensure_ascii=False)}

            User Memory (extract values from here):
            {json.dumps(memory_chain, indent=2, ensure_ascii=False)}

            CRITICAL REQUIREMENTS:
            1. At least one parameter values MUST come from the memory above (Explicit or Inferred).
            2. Match parameter types exactly (string/integer/number/boolean/array/object).
            3. For nested structures (arrays of objects), follow the schema's items/properties definitions.
            4. All required parameters must be filled.
            5. PREFER ORIGINAL TEXT: If a parameter can be found in memory, USE THE EXACT ORIGINAL TEXT from the memory.
            
            6. PARAMETER SOURCES:
               - Memory (Explicit): Value appears verbatim in memory (e.g., "ID is 123" -> id="123").
               - Memory (Inferred): Value is inferred from context (e.g., "visiting Eiffel Tower" -> city="Paris").
               - Schema (Default): Value comes from the tool schema's default value if not found in memory.
               - Multi-hop: Parameters come from multiple different facts.

            7. GROUNDING INFO:
               - For each extracted parameter, specify:
                 * "source_text": The text in memory used for extraction (or "default value").
                 * "type": "explicit", "inferred", or "default".

            8. SOURCE IDENTIFICATION:
               - Identify which specific memory items (by their 'source_id') were necessary to derive the parameters.
               - List ALL source_ids that contributed to the parameters (including inferred ones).
               - If a parameter comes from a default value, do not include a source_id for it unless it was verified against memory.

            Output JSON (follow this structure):
            {json.dumps(example_output, indent=2, ensure_ascii=False)}
            """
        
        try:
            response = self.llm_client.chat_completion(
                model=self.model, temperature=0,
                messages=[{"role": "system", "content": "You are a precise parameter extraction system. Support explicit extraction, semantic inference, and default values."}, 
                         {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = safe_json_loads(response.choices[0].message.content, {})
            
            arguments = result.get("arguments", {})
            grounding_info = result.get("grounding_info", {})
            relevant_source_ids = result.get("relevant_source_ids", [])
            
            # 过滤并验证 source_ids
            valid_source_ids = {item.get("source_id") for item in memory_chain if item.get("source_id")}
            verified_source_ids = [sid for sid in relevant_source_ids if sid in valid_source_ids]
            result["relevant_source_ids"] = verified_source_ids
            
            # 构建用于验证的 memory 子集
            validation_memory_chain = [item for item in memory_chain if item.get("source_id") in verified_source_ids]
            # 生成 validation text (用于 inference 验证)
            validation_memory_text = json.dumps(validation_memory_chain, ensure_ascii=False, indent=2)
            
            if not arguments:
                LOGGER.debug("参数为空")
                return None
            
            # 验证 1: Schema校验
            is_valid_schema, schema_error = self.schema_validator.validate_tool_call(tool, arguments)
            if not is_valid_schema:
                LOGGER.debug(f"Schema校验失败: {schema_error}")
                return None
            
            # 分离 explicit, inferred 和 default 参数
            explicit_params = {}
            inferred_params = {}
            default_params = {}
            
            for param, info in grounding_info.items():
                if param not in arguments: continue
                
                p_type = info.get("type", "explicit")
                if p_type == "inferred":
                    inferred_params[param] = (arguments[param], info.get("source_text", ""))
                elif p_type == "default":
                    default_params[param] = arguments[param]
                else:
                    explicit_params[param] = arguments[param]

            grounded_params = []
            param_to_memory = {}
            param_modes = {}

            # 验证 2a: Explicit 参数 (使用 MemoryGroundingValidator)
            if explicit_params:
                is_grounded, g_params, p_map, p_modes = self.grounding_validator.validate_grounding(
                    explicit_params, validation_memory_chain, [], 
                    llm_grounded_params=list(explicit_params.keys()),
                    mode=self.grounding_mode
                )
                if is_grounded or g_params:
                    grounded_params.extend(g_params)
                    param_to_memory.update(p_map)
                    param_modes.update(p_modes)
            
            # 验证 2b: Inferred 参数 (使用 LLM 验证)
            if inferred_params:
                for param, (val, source) in inferred_params.items():
                    if self._verify_inference(param, val, source, validation_memory_text):
                        grounded_params.append(param)
                        param_modes[param] = "inferred"
                        # 尝试找到对应的 memory id
                        for item in validation_memory_chain:
                            if item.get("fact") and source in item.get("fact"):
                                param_to_memory[param] = item.get("id")
                                break
                        if param not in param_to_memory:
                             param_to_memory[param] = "inferred_source"

            # 验证 2c: Default 参数 (直接接受，不计入 grounded_params，但标记为 default)
            if default_params:
                for param in default_params:
                    param_modes[param] = "default"
                    param_to_memory[param] = "schema_default"

            # 检查是否满足必填参数要求
            # 至少有一个参数必须来自 Memory (Explicit 或 Inferred)
            if not grounded_params:
                LOGGER.debug("没有参数锚定到记忆 (All defaults or failed grounding)")
                return None
            
            LOGGER.debug(f"工具调用构造成功: {result.get('name')}, 锚定参数: {grounded_params}, 模式: {param_modes}")
            
            # 增强复杂度元数据（用于难度分级）
            result = self._enrich_with_complexity_metadata(
                result, 
                grounding_info, 
                verified_source_ids,
                validation_memory_chain
            )
            
            return result
            
        except Exception as e:
            LOGGER.error(f"工具调用构造失败: {e}")
            return None

    def _verify_inference(self, param: str, value: Any, source_text: str, full_memory: str) -> bool:
        """验证推断参数的合理性"""
        prompt = f"""Verify if the parameter inference is logically sound based on the user memory.

        User Memory:
        {full_memory}

        Inference Claim:
        - Parameter: {param}
        - Inferred Value: {value}
        - Based on text: "{source_text}"

        Is this inference correct and logical? (e.g., "Eiffel Tower" -> "Paris" is YES. "Apple" -> "Banana" is NO).
        
        Output JSON:
        {{
            "is_valid": true/false,
            "reason": "explanation"
        }}
        """
        try:
             response = self.llm_client.chat_completion(
                model=self.model, temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
             res = safe_json_loads(response.choices[0].message.content, {})
             return res.get("is_valid", False)
        except:
            return False
    
    def _enrich_with_complexity_metadata(
        self, 
        tool_call: Dict, 
        grounding_info: Dict,
        relevant_source_ids: List[str],
        memory_chain: List[Dict]
    ) -> Dict:
        """为工具调用增加复杂度元数据（用于难度分级 D2: Cognitive Depth）
        
        根据 '难度分级思路.md' 的 D2 维度标注，采用简化策略：
        - L1: Atomic-Explicit (单源 + 显式)
        - L2: Atomic-Inferred (单源 + 推断)
        - L3: Composite-Aggregation (多源 + 全显式)
        - L4: Composite-Deduction (多源 + 存在推断)
        """
        
        # 1. 统计参数类型
        explicit_count = 0
        inferred_count = 0
        default_count = 0
        
        for param, info in grounding_info.items():
            p_type = info.get("type", "explicit")
            if p_type == "inferred":
                inferred_count += 1
            elif p_type == "default":
                default_count += 1
            else:
                explicit_count += 1
                
        # 2. 统计 Source 数量
        # 直接使用传入的 relevant_source_ids，不再尝试将每个参数精确映射到 source_id
        unique_source_count = len(relevant_source_ids)
        
        # 3. 判定认知级别（D2 Cognitive Level）
        # 逻辑：
        # - source > 1: Composite (L3 or L4)
        #   - if any inferred param -> L4 (Deduction/Intersection)
        #   - else (all explicit/default) -> L3 (Aggregation)
        # - source <= 1: Atomic (L1 or L2)
        #   - if any inferred param -> L2 (Inferred)
        #   - else -> L1 (Explicit)

        if unique_source_count > 1:
            # Composite 级别
            if inferred_count > 0:
                cognitive_level = "L4"
                reasoning_type = "Composite-Deduction"
            else:
                cognitive_level = "L3"
                reasoning_type = "Composite-Aggregation"
        else:
            # Atomic 级别
            if inferred_count > 0:
                cognitive_level = "L2"
                reasoning_type = "Atomic-Inferred"
            else:
                cognitive_level = "L1"
                reasoning_type = "Atomic-Explicit"
        
        # 4. 构建并添加复杂度元数据
        complexity_metadata = {
            "cognitive_level": cognitive_level,
            "reasoning_type": reasoning_type,
            "total_params": len(grounding_info),
            "explicit_params": explicit_count,
            "inferred_params": inferred_count,
            "default_params": default_count,
            "unique_sources": unique_source_count,
            "relevant_source_ids": relevant_source_ids,
            "is_composite": unique_source_count > 1,
            "requires_deduction": (unique_source_count > 1 and inferred_count > 0)
        }
        
        tool_call["complexity_metadata"] = complexity_metadata
        
        LOGGER.debug(f"复杂度标注: {reasoning_type} (L{cognitive_level[-1]}), "
                    f"参数: {explicit_count}显式/{inferred_count}推理/{default_count}默认, "
                    f"来源: {unique_source_count}个source")
        
        return tool_call


class ReverseQueryGenerator:
    """逆向Query生成"""
    def __init__(self, model: str):
        self.model = model
        self.llm_client = LLMClient()
    
    def generate(self, tool_call: Dict, memory_chain: List[Dict]) -> Optional[str]:
        memory_text = "\n".join([item.get("fact", "") for item in memory_chain if item.get("fact")])
        # grounded = tool_call.get("grounded_params", [])
        args = tool_call.get("arguments", {})
        
        prompt = f"""You are generating a challenging, memory-dependent user query for testing an AI agent's long-term memory capabilities.

            Context:
            - The assistant has access to various tools and can call them to complete user requests.
            - You are given a specific tool call that the assistant will make, along with the user's memory context.
            - Your task is to generate a natural user query that would REQUIRE the assistant to call exactly this tool, but crucially, the user MUST OMIT key details that are already present in their memory/history.

            Tool Call (the assistant will execute this):
            {json.dumps(tool_call, indent=2, ensure_ascii=False)}

            User's Memory Context (History/Preferences):
            {memory_text}

            Requirements for the generated query:

            1. MUST BE TOOL-DEPENDENT
            - The query CANNOT be answered using the model's parametric knowledge alone.
            - It MUST require accessing external data or performing operations.

            2. STRICTLY IMPLICIT (NO PARAMETER LEAKAGE)
            - You MUST OMIT parameter values that are already established in the memory context.
            - If the memory contains the destination "Dallas", the query MUST NOT say "Dallas". It should say "my destination" or "there".
            - If the memory contains the date "March 13th", the query MUST NOT say "March 13th". It should say "that day" or "the date we discussed".
            - The query MUST be ambiguous without the memory, but clear with the memory.

            3. PROVIDE DOMAIN CONTEXT (CRITICAL)
            - While avoiding specific parameter values, you MUST include enough semantic context (topic, category, or action nature) so the user knows WHICH memory thread is being referred to.
            - AVOID purely generic pronouns like "it", "that", "those numbers" without any category.
            - BAD: "Can you check the numbers?" (Too vague, could be anything)
            - GOOD: "Can you check the atmospheric numbers?" (Better, domain is clear, but specific gas is hidden)
            - GOOD: "How is the air quality data looking?" (Good, implies the topic without stating "Nitrous Oxide")
            - BAD: "Book it." (Too vague)
            - GOOD: "Book that flight." (Better, domain is travel)
            
            4. REFLECT TOOL SPECIFICS (DISAMBIGUATION)
            - If the tool requires a specific type of identifier (e.g., SecUID vs Username), the query should imply that specific type without stating the value.
            - Example: If the tool uses 'SecUID', the query should say "using the secure ID I gave you" rather than just "my account".
            - This ensures the query logically leads to the specific tool selected.
            
            5. NATURAL AND CONVERSATIONAL
            - Use phrases like "as usual", "like I said before", "for my trip", "book it", "check that thing".
            - Make it sound like a continuing conversation or a user with a long history.

            6. NO SYSTEM INTERNALS
            - Do NOT mention "tool", "function", "API", "call", "parameter".

            Output Format:
            Return ONLY a valid JSON object with one field:

            {{
            "query": "the generated user query here"
            }}

            Do not include any explanations, comments, or text outside this JSON object.
            """
        
        try:
            response = self.llm_client.chat_completion(
                model=self.model, temperature=0,  # 使用temperature=0以保证完全可复现
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, timeout=30
            )
            result = safe_json_loads(response.choices[0].message.content, {})
            return result.get("query")
        except Exception as e:
            LOGGER.error(f"Query生成失败: {e}")
            return None

def check_argument_leakage(query: str, tool_call: Dict) -> bool:
    """参数泄露检查 - 智能版
    
    仅检查那些「标记为来自记忆」的参数是否泄露。
    默认参数或非记忆参数允许出现在 Query 中。
    """
    query_lower = query.lower()
    args = tool_call.get("arguments", {})
    # 获取参数的锚定来源信息
    grounding_info = tool_call.get("grounding_info", {}) 
    
    for param_name, param_value in args.items():
        # === 核心优化逻辑 ===
        # 1. 获取该参数的来源类型
        p_info = grounding_info.get(param_name, {})
        source_type = p_info.get("type", "unknown")

        # 2. 白名单过滤：只有 "explicit" 或 "inferred" 的参数才严禁泄露
        # 如果参数是 "default"，或者没有来源标记，我们假设它不需要被隐藏
        if source_type not in ["explicit", "inferred"]:
            continue

        # === 以下是针对记忆参数的严格检查 ===
        value_str = str(param_value).strip()
        
        # 跳过过短的值（避免误杀如 "1", "US", "on"）
        if len(value_str) < 2: 
            continue
        
        # [数值检测]
        try:
            num_val = float(value_str)
            # 检查数字是否直接出现在query中 (例如参数是 123456)
            if str(int(num_val)) in query or value_str in query:
                LOGGER.debug(f"检测到记忆参数(数值)泄露: {param_name}={value_str} (Source: {source_type})")
                return True
        except ValueError:
            pass
        
        # [字符串检测]
        value_lower = value_str.lower()
        
        # A. 直接整词匹配
        # 使用边界符 \b 避免匹配到单词的一部分 (如 value="cat" 匹配到 "category")
        pattern = r'\b' + re.escape(value_lower) + r'\b'
        if re.search(pattern, query_lower):
            LOGGER.debug(f"检测到记忆参数(整词)泄露: {param_name}={value_str} (Source: {source_type})")
            return True

        # B. 关键词拆分检查 (针对长复合词)
        # 只有当参数值较长时才拆分检查，避免把简单词拆散后误杀
        if len(value_lower) > 4:
            # 分隔符：逗号、空格、下划线、连字符
            keywords = re.split(r'[,\s_\-]+', value_lower)
            # 过滤掉短词和停用词
            stopwords = {"the", "and", "for", "with", "from", "that", "this", "what", "how"}
            keywords = [k.strip() for k in keywords if len(k.strip()) >= 3 and k.strip() not in stopwords]
            
            for kw in keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, query_lower):
                    LOGGER.debug(f"检测到记忆参数(关键词 '{kw}')泄露: {param_name}={value_str} (Source: {source_type})")
                    return True
        
        # C. 特殊格式检查 (Email, URL, ID)
        if any(char in value_str for char in ['@', 'http', '://']):
             if value_str in query or value_lower in query_lower:
                LOGGER.debug(f"检测到记忆参数(特殊格式)泄露: {param_name}={value_str}")
                return True
    
    return False

class LLMQualityEvaluator:
    """基于LLM的质量评估器 - 作为判题器而非出题人
    
    类似 StableToolBench 的做法，用LLM评估生成的QA质量
    """
    def __init__(self, model: str):
        self.model = model
        self.llm_client = LLMClient()
    
    def evaluate(self, query: str, tool_call: Dict, memory_chain: List[Dict]) -> Tuple[bool, str]:
        """评估QA质量（增强版：整合memory dependency检查）
        
        Returns:
            (is_acceptable, rejection_reason)
        """
        memory_text = "\n".join([f"- {item.get('fact', '')}" for item in memory_chain if item.get('fact')])
        
        prompt = f"""You are a quality evaluator for tool-calling QA pairs.

            Evaluate whether the following QA pair is valid and high-quality.

            User Memory Context:
            {memory_text}

            User Query:
            {query}

            Tool Call:
            Tool: {tool_call.get('name')}
            Arguments: {json.dumps(tool_call.get('arguments', {}), ensure_ascii=False)}

            Evaluation Criteria:
            1. CLARITY: Is the query clear and specific enough?
            2. INFORMATION SUFFICIENCY: Does the query provide enough context (without leaking parameter values)?
            3. TOOL APPROPRIATENESS: Is the selected tool reasonable for the query?
            4. PARAMETER REASONABLENESS: Are the parameters consistent with the query intent?
            5. MEMORY DEPENDENCY (CRITICAL - MUST CHECK STRICTLY):
               - Cover the "User Memory Context" and look ONLY at the "User Query".
               - Can you infer the tool parameters based solely on the query?
               - If the query says "Book a ticket to Dallas", and the tool arg is "Dallas", you MUST REJECT it.
               - If the query says "Book a ticket for my trip", and you CANNOT know it's "Dallas" without memory, then ACCEPT it.
               - If the query says "3 people", and the tool arg is "3", you MUST REJECT it.
               - If the query contains ANY specific values (dates, names, locations, numbers) that map directly to tool arguments, REJECT it.
               - REJECT REASON format: "solvable_without_memory: <explanation>"

            Common Issues to Reject:
            - Query leaks specific parameter values (names, dates, locations, numbers)
            - Task can be solved without memory
            - Query is too vague or ambiguous (without memory context)
            - Tool selection doesn't match the query intent

            IMPORTANT: If the query provides enough information to fill parameters without needing the memory context,
            you MUST reject it with rejection_reason starting with "solvable_without_memory:".

            Output JSON:
            {{
                "status": "ACCEPT" or "REJECT",
                "rejection_reason": "specific reason if rejected (use 'solvable_without_memory:' prefix when applicable), empty if accepted",
                "quality_score": 1-5,
                "reasoning": "brief explanation"
            }}
            """
        
        try:
            response = self.llm_client.chat_completion(
                model=self.model, temperature=0,
                messages=[{"role": "system", "content": "You are a strict quality evaluator. Only accept high-quality, memory-dependent QA pairs."}, 
                         {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, timeout=30
            )
            result = safe_json_loads(response.choices[0].message.content, {})
            
            status = result.get("status", "REJECT").upper()
            rejection_reason = result.get("rejection_reason", "Unknown reason")
            
            if status == "ACCEPT":
                return True, ""
            else:
                LOGGER.debug(f"LLM评估器拒绝: {rejection_reason}")
                return False, rejection_reason
        
        except Exception as e:
            LOGGER.error(f"LLM评估失败: {e}")
            # 如果评估失败，默认拒绝
            return False, "Evaluation failed"

class MemoryDrivenQAConstructor:
    """QA构造器 - 增强版：有约束的生成 + 严格过滤 + 「硬+软」两级匹配"""
    def __init__(self, memory_path: Path, tool_pool_path: Path, output_path: Path, llm_model: str,
                 max_event_groups: Optional[int] = None, enable_llm_evaluation: bool = True, 
                 max_workers: int = 5, grounding_mode: str = "strict"):
        self.output_path = output_path
        self.max_event_groups = max_event_groups
        self.enable_llm_evaluation = enable_llm_evaluation
        self.max_workers = max_workers
        self.grounding_mode = grounding_mode  # "strict" 或 "hybrid"
        self.stats = defaultdict(int)
        self.stats_lock = threading.Lock()
        self.file_lock = threading.Lock()  # 文件写入锁
        
        LOGGER.info(f"加载记忆数据: {memory_path}")
        # 加载CSV文件并处理记忆演化链
        self.event_groups = self._load_memory_from_csv(memory_path)
        
        LOGGER.info(f"加载工具池: {tool_pool_path}")
        # 初始化LLM客户端
        llm_client = LLMClient()
        # 初始化工具池（启用混合语义检索）
        self.tool_pool = ToolPoolManager(
            tool_pool_path, 
            enable_advanced_retrieval=True  # 启用混合检索（Dense + Sparse）
        )
        
        # 初始化所有组件
        self.mapper = MemoryToToolMapper(llm_model, self.tool_pool)
        self.constructor = ToolCallConstructor(llm_model, grounding_mode=grounding_mode)
        self.query_gen = ReverseQueryGenerator(llm_model)
        self.llm_evaluator = LLMQualityEvaluator(llm_model) if enable_llm_evaluation else None
        
        LOGGER.info(f"配置: grounding_mode={grounding_mode}, LLM评估(含 memory dependency检查)={enable_llm_evaluation}, 并发数={max_workers}")
    
    def _load_memory_from_csv(self, csv_path: Path) -> List[Dict]:
        """从CSV文件加载记忆演化链数据
        
        处理逻辑：
        1. 读取CSV文件
        2. 解析sorted_source_ids, discarded_source_ids, original_facts
        3. 过滤掉discarded_source_ids中的内容
        4. 构建记忆演化链
        """
        LOGGER.info(f"从CSV加载记忆数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        event_groups = []
        for idx, row in df.iterrows():
            try:
                # 解析字符串形式的列表和JSON
                sorted_source_ids = ast.literal_eval(row['sorted_source_ids'])
                discarded_source_ids = ast.literal_eval(row['discarded_source_ids'])
                original_facts = json.loads(row['original_facts'])
                
                # 转换为集合以便快速查找
                discarded_set = set(discarded_source_ids)
                
                # 过滤掉被丢弃的source_ids
                valid_source_ids = [sid for sid in sorted_source_ids if sid not in discarded_set]
                
                # 过滤original_facts，只保留有效的source_id
                valid_facts = [
                    fact for fact in original_facts 
                    if fact.get('source_id') not in discarded_set
                ]
                
                # 按照sorted_source_ids的顺序重新排列facts
                # 创建source_id到fact的映射
                fact_map = {fact.get('source_id'): fact for fact in valid_facts}
                
                # 按照sorted_source_ids的顺序构建演化链
                evolution_chain = []
                for source_id in valid_source_ids:
                    if source_id in fact_map:
                        fact_item = fact_map[source_id]
                        # 构建标准化的记忆项
                        evolution_chain.append({
                            "attribute": fact_item.get('attribute', ''),
                            "source_id": fact_item.get('source_id'),
                            "fact": fact_item.get('facts', ''),
                            "source_text": fact_item.get('source_text', ''),
                        })
                
                # 只有当演化链不为空时才添加
                if evolution_chain:
                    event_groups.append({
                        "thread_id": f"event_group_{idx}",
                        "thread_topic": row['attribute_group'],
                        "narrative_summary": row['narrative_summary'],
                        "evolution_chain": evolution_chain,
                    })
                    
            except Exception as e:
                LOGGER.warning(f"处理第 {idx} 行时出错: {e}")
                continue
        
        LOGGER.info(f"成功加载 {len(event_groups)} 个记忆演化链")
        return event_groups
    
    def run(self):
        # 1. 加载断点信息
        max_processed_index = -1
        if self.output_path.exists():
            LOGGER.info(f"发现已有输出文件 {self.output_path}，正在加载断点信息...")
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            item = json.loads(line)
                            idx = item.get("source_index")
                            if idx is not None:
                                max_processed_index = max(max_processed_index, idx)
                        except:
                            continue
                LOGGER.info(f"检测到最大已处理索引: {max_processed_index} (将跳过 <= 此索引的任务)")
            except Exception as e:
                LOGGER.warning(f"加载断点信息失败: {e}")

        # 2. 收集并过滤任务
        LOGGER.info(f"开始收集任务...")
        all_tasks = [] # (index, thread)
        
        # 展开所有 event_groups 为 threads，并保持全局索引
        global_idx = 0
        for event_group in self.event_groups[:self.max_event_groups]:
            threads = event_group if isinstance(event_group, list) else [event_group]
            for thread in threads:
                if global_idx > max_processed_index:
                    all_tasks.append((global_idx, thread))
                global_idx += 1
        
        LOGGER.info(f"总任务数: {global_idx}, 剩余待处理: {len(all_tasks)}")

        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 3. 并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务，传入 (index, thread)
            future_to_task = {
                executor.submit(self._process_thread_safe, thread, idx): idx 
                for idx, thread in all_tasks
            }
            
            # 使用tqdm显示进度
            for future in tqdm(as_completed(future_to_task), total=len(all_tasks), desc="Processing\n"):
                try:
                    qa = future.result()
                    if qa:
                        self._append_result(qa)
                except Exception as e:
                    LOGGER.error(f"任务处理异常: {e}")
        
        self._print_stats()
    
    def _process_thread_safe(self, thread: Dict, index: int) -> Optional[Dict]:
        """线程安全的处理函数"""
        with self.stats_lock:
            self.stats['total_attempted'] += 1
        
        qa = self._process_thread(thread, index)
        
        if qa:
            with self.stats_lock:
                self.stats['total_accepted'] += 1
        
        return qa
    
    def _process_thread(self, thread: Dict, index: int) -> Optional[Dict]:
        """处理单个记忆线程"""
        thread_id = thread.get("thread_id")
        evolution_chain = thread.get("evolution_chain", [])
        
        if not evolution_chain:
            with self.stats_lock:
                self.stats['rejected_no_evolution'] += 1
            return None
        
        # Step 1: 工具选择（规则预选 + LLM离散选择）
        tool_selection = self.mapper.analyze_and_select_tool(evolution_chain)
        if not tool_selection:
            print("Step1: No tool selected")
            with self.stats_lock:
                self.stats['rejected_no_tool'] += 1
            return None
        
        tool = tool_selection["tool"]
        tool_selection_metadata = tool_selection.get("tool_selection_metadata", {})
        
        # Step 2: 构造工具调用（Schema约束 + 记忆锚定）
        tool_call = self.constructor.construct_call(tool, evolution_chain)
        if not tool_call:
            print("Step2: Construction failed")
            with self.stats_lock:
                self.stats['rejected_construction_failed'] += 1
            return None

        # 提取 relevant_source_ids 并过滤 memory_chain
        relevant_source_ids = tool_call.get("relevant_source_ids", [])
        relevant_memory_chain = [
            item for item in evolution_chain 
            if item.get("source_id") in relevant_source_ids
        ]

        # Step 3: 逆向生成Query
        masked_query = self.query_gen.generate(tool_call, relevant_memory_chain)
        if not masked_query:
            print("Step3: Query generation failed")
            with self.stats_lock:
                self.stats['rejected_query_gen_failed'] += 1
            return None
        
        # Step 4: 参数泄露检查（增强版）
        if check_argument_leakage(masked_query, tool_call):
            print("Step4: Argument leakage detected")
            with self.stats_lock:
                self.stats['rejected_argument_leak'] += 1
            return None
    
        # Step 5: LLM质量评估（整合memory dependency检查）
        if self.enable_llm_evaluation and self.llm_evaluator:
            is_acceptable, llm_rejection = self.llm_evaluator.evaluate(masked_query, tool_call, relevant_memory_chain)
            if not is_acceptable:
                print("Step5: LLM evaluation failed")
                with self.stats_lock:
                    self.stats['rejected_llm_evaluation'] += 1
                    self.stats[f'llm_rejection_{llm_rejection[:30]}'] += 1
                LOGGER.debug(f"LLM评估拒绝 [Thread: {thread_id}]: {llm_rejection}")
                return None
        
        # 构造最终样本
        return {
            "thread_id": thread_id,
            "thread_topic": thread.get("thread_topic", ""),
            "evolution_chain": relevant_memory_chain,
            "qa_pair": {
                "query": masked_query,
                "tool_calls": [tool_call],
            },
            "target_tool_schema": tool,
            "tool_selection_reasoning": tool_selection_metadata.get("selection_reasoning", ""),
            "source_records": relevant_source_ids,
            "source_index": index,
        }
    
    def _append_result(self, result: Dict):
        """追加写入单条结果（线程安全）"""
        try:
            with self.file_lock:
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            LOGGER.error(f"写入结果失败: {e}")
    
    def _print_stats(self):
        LOGGER.info("=" * 80)
        LOGGER.info("处理统计 - 增强版 Pipeline")
        LOGGER.info("=" * 80)
        LOGGER.info(f"尝试总数: {self.stats['total_attempted']}")
        LOGGER.info(f"成功接受: {self.stats['total_accepted']}")
        
        # 计算接受率
        if self.stats['total_attempted'] > 0:
            accept_rate = (self.stats['total_accepted'] / self.stats['total_attempted']) * 100
            LOGGER.info(f"接受率: {accept_rate:.2f}%")
        
        LOGGER.info("-" * 80)
        LOGGER.info("拒绝原因统计 (分阶段 - 精简版):")
        LOGGER.info(f"  [Stage 1] 无演化链: {self.stats['rejected_no_evolution']}")
        LOGGER.info(f"  [Stage 2] 无合适工具 (规则预选 + LLM选择): {self.stats['rejected_no_tool']}")
        LOGGER.info(f"  [Stage 3] 构造失败 (Schema + 记忆锚定): {self.stats['rejected_construction_failed']}")
        LOGGER.info(f"  [Stage 4] Query生成失败: {self.stats['rejected_query_gen_failed']}")
        LOGGER.info(f"  [Stage 5] 参数泄露检查: {self.stats['rejected_argument_leak']}")
        LOGGER.info(f"  [Stage 6] 基础质量检查 (结构化规则): {self.stats['rejected_basic_quality']}")
        LOGGER.info(f"  [Stage 7] LLM评估 (memory dependency + 质量): {self.stats['rejected_llm_evaluation']}")
        
        # 显示基础质量检查的具体拒绝原因
        basic_quality_rejections = {k: v for k, v in self.stats.items() if k.startswith('basic_quality_')}
        if basic_quality_rejections:
            LOGGER.info("\n  基础质量检查详细原因:")
            for reason, count in sorted(basic_quality_rejections.items(), key=lambda x: x[1], reverse=True):
                reason_text = reason.replace('basic_quality_', '')
                LOGGER.info(f"    - {reason_text}: {count}")
        
        # 显示LLM评估的具体拒绝原因
        llm_rejections = {k: v for k, v in self.stats.items() if k.startswith('llm_rejection_')}
        if llm_rejections:
            LOGGER.info("\n  LLM评估拒绝详细原因 (Top 10):")
            for reason, count in sorted(llm_rejections.items(), key=lambda x: x[1], reverse=True)[:10]:
                reason_text = reason.replace('llm_rejection_', '')
                LOGGER.info(f"    - {reason_text}: {count}")
        
        LOGGER.info("-" * 80)
        LOGGER.info("难度分布:")
        total_by_difficulty = sum(self.stats.get(f'difficulty_level_{level}', 0) for level in [1, 2, 3])
        for level in [1, 2, 3]:
            count = self.stats.get(f'difficulty_level_{level}', 0)
            percentage = (count / total_by_difficulty * 100) if total_by_difficulty > 0 else 0
            LOGGER.info(f"  Level {level}: {count} ({percentage:.1f}%)")
        
        LOGGER.info("-" * 80)
        LOGGER.info("Grounding 质量分布:")
        strict_gold_count = self.stats.get('grounding_strict_gold', 0)
        soft_gold_count = self.stats.get('grounding_soft_gold', 0)
        total_grounding = strict_gold_count + soft_gold_count
        if total_grounding > 0:
            LOGGER.info(f"  Strict Gold (100% 精确匹配): {strict_gold_count} ({strict_gold_count/total_grounding*100:.1f}%)")
            LOGGER.info(f"  Soft Gold (含软匹配): {soft_gold_count} ({soft_gold_count/total_grounding*100:.1f}%)")
        else:
            LOGGER.info("  无样本")
    
    def _save(self, results: List[Dict]):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"保存 {len(results)} 个样本到 {self.output_path}")
        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Memory-driven QA Construction with Constrained Generation (Optimized - 4 LLM calls per sample)",
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("--memory-path", type=Path, default=Path("processed_data/conflict_evolutions.csv"),
                       help="路径到记忆演化数据 (CSV格式)")
    parser.add_argument("--tool-pool-path", type=Path, default=Path("tool_api_pool.json"),
                       help="路径到静态工具池 (e.g., ToolACE)")
    parser.add_argument("--llm-model", type=str, default="moonshotai/Kimi-K2-Instruct-0905",help="生成使用的 LLM 模型")
    parser.add_argument("--max-event-groups", type=int, default=None ,
                       help="最大处理的事件组数 (用于调试)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="并行处理的线程数")
    parser.add_argument("--disable-llm-evaluation", action="store_true",
                       help="禁用 LLM 质量评估 (含 memory dependency 检查)")
    parser.add_argument("--grounding-mode", type=str, default="hybrid", choices=["strict", "hybrid"],
                       help="参数锚定模式: strict (仅精确匹配) 或 hybrid (精确+软匹配)")
    
    args = parser.parse_args()
    args.output = Path(f"processed_data/memory_driven_qa_{args.llm_model.split('/')[-1]}_v2.jsonl")
    
    constructor = MemoryDrivenQAConstructor(
        memory_path=args.memory_path,
        tool_pool_path=args.tool_pool_path,
        output_path=args.output,
        llm_model=args.llm_model,
        max_event_groups=args.max_event_groups,
        enable_llm_evaluation=not args.disable_llm_evaluation,
        max_workers=args.max_workers,
        grounding_mode=args.grounding_mode,
    )
    constructor.run()

if __name__ == "__main__":
    main()
