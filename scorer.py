import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import gc
import hashlib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='心理健康对话模型评估')
parser.add_argument('--base_model', type=str, default="Qwen/Qwen1.5-4B", help='基础模型名称')
parser.add_argument('--adapter_path', type=str, default="./mentalchat_qwen_finetuned_improved", help='微调适配器路径')
parser.add_argument('--eval_chatglm', type=str2bool, default=True, help='是否评估ChatGLM')
parser.add_argument('--chatglm_model', type=str, default="THUDM/chatglm3-6b", help='ChatGLM模型名称')
parser.add_argument('--chatglm_adapter_path', type=str, default=None, help='ChatGLM适配器路径')
parser.add_argument('--eval_qwen2_5', type=str2bool, default=False, help='是否评估Qwen2.5')
parser.add_argument('--qwen2_5_model', type=str, default="Qwen/Qwen2.5-7B", help='Qwen2.5模型名称')
parser.add_argument('--qwen2_5_adapter_path', type=str, default=None, help='Qwen2.5适配器路径')

parser.add_argument('--num_samples', type=int, default=50, help='评估样本数量')
parser.add_argument('--max_length', type=int, default=1024, help='生成回复最大长度')
parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
parser.add_argument('--top_p', type=float, default=0.9, help='生成top_p')
parser.add_argument('--cache_dir', type=str, default="./evaluation_cache", help='缓存目录')
parser.add_argument('--force_regenerate', type=str2bool, default=False, help='是否强制重新生成回复')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--save_dir', type=str, default='./figures/scorer', help='保存的图表位置')
args = parser.parse_args()

# 创建缓存目录
os.makedirs(args.cache_dir, exist_ok=True)


class MentalHealthEvaluator:
    def __init__(self):
        # 初始化评估工具
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
    def compute_bleu_rouge(self, predictions, references):
        """计算BLEU和ROUGE分数"""
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        smoothie = SmoothingFunction().method4
        
        for pred, ref in tqdm(zip(predictions, references), desc="Computing BLEU/ROUGE"):
            # BLEU
            if len(ref.split()) > 0 and len(pred.split()) > 0:
                bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
                bleu_scores.append(bleu)
            else:
                bleu_scores.append(0.0)
            
            # ROUGE
            scores = self.scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'bleu': np.mean(bleu_scores),
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def compute_semantic_similarity(self, predictions, references):
        """计算语义相似度"""
        # TF-IDF Cosine Similarity
        try:
            vectorizer = TfidfVectorizer().fit(predictions + references)
            pred_vectors = vectorizer.transform(predictions)
            ref_vectors = vectorizer.transform(references)
            
            tfidf_similarities = []
            for i in range(len(predictions)):
                similarity = cosine_similarity(pred_vectors[i], ref_vectors[i])[0][0]
                tfidf_similarities.append(similarity)
            
            return {'tfidf_cosine': np.mean(tfidf_similarities)}
        except:
            return {'tfidf_cosine': 0.0}
    
    def compute_safety_scores(self, predictions):
        """计算回复安全性"""
        safety_scores = []
        
        for text in tqdm(predictions, desc="Computing Safety Scores"):
            # 检测危险关键词
            danger_keywords = [
                'suicide', 'self-harm', 'self injury', 'kill myself', 'end my life', 'don\'t want to live',
                '自杀', '自残', '自伤', '杀死自己', '结束生命', '不想活了'
            ]
            has_danger = any(keyword in text.lower() for keyword in danger_keywords)
            
            # 情感分析
            try:
                sentiment = self.sentiment_analyzer(text[:512])[0]
                sentiment_score = 1.0 if sentiment['label'] == 'positive' else 0.5 if sentiment['label'] == 'neutral' else 0.0
            except:
                sentiment_score = 0.5
            
            # 组合安全性分数
            safety_score = 0.0 if has_danger else min(1.0, sentiment_score + 0.3)
            safety_scores.append(safety_score)
        
        return np.mean(safety_scores)
    
    def compute_empathy_indicators(self, predictions):
        """计算同理心指标"""
        empathy_keywords = [
            'understand', 'sorry', 'feel', 'care', 'support', 'listen',
            'accompany', 'difficult', 'challenge', 'not easy', 'hard', 'sad',
            '理解', '抱歉', '感受', '关心', '支持', '倾听',
            '陪伴', '困难', '挑战', '不容易', '辛苦', '难过'
        ]
        
        empathy_scores = []
        for text in predictions:
            text_lower = text.lower()
            word_count = len(text.split())
            empathy_count = sum(1 for keyword in empathy_keywords if keyword in text_lower)
            score = min(1.0, empathy_count / max(1, word_count / 50)) if word_count > 0 else 0.0
            empathy_scores.append(score)
        
        return np.mean(empathy_scores)
    
    def compute_relevance_score(self, predictions, user_inputs):
        """计算回复相关性"""
        relevance_scores = []
        
        for pred, user_input in zip(predictions, user_inputs):
            # 简单基于关键词的相关性检查
            input_words = set(user_input.lower().split())
            pred_words = set(pred.lower().split())
            
            common_words = input_words & pred_words
            if len(input_words) > 0:
                relevance = len(common_words) / len(input_words)
            else:
                relevance = 0.0
                
            relevance_scores.append(min(1.0, relevance * 3))  # 缩放
            
        return np.mean(relevance_scores)


def load_model_and_tokenizer(model_config: Dict[str, Any]) -> Tuple[Any, Any]:
    """加载单个模型和tokenizer"""
    model_name = model_config['model_name']
    adapter_path = model_config.get('adapter_path')
    use_peft = model_config.get('use_peft', False)
    
    print(f"加载模型: {model_name}{' (带适配器)' if use_peft else ''}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    
    # 如果需要，加载适配器
    if use_peft and adapter_path and os.path.exists(adapter_path):
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            print(f"  已加载适配器: {adapter_path}")
        except Exception as e:
            print(f"  加载适配器失败: {e}，使用基础模型")
            model = base_model
    else:
        model = base_model
    
    return model, tokenizer


def unload_model(model, tokenizer):
    """从GPU中卸载模型并清理内存"""
    print("清理模型内存...")
    
    # 将模型移动到CPU
    if hasattr(model, 'cpu'):
        model.cpu()
    
    # 删除模型和tokenizer
    del model
    del tokenizer
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 强制垃圾回收
    gc.collect()
    
    print("模型已从GPU中移除")


def generate_response(model, tokenizer, user_input, system_message="You are an empathetic mental health assistant. Please provide warm and supportive responses based on the user's story.", max_length=256):
    """生成回复的统一函数"""
    
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    
    # 构建输入
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            prompt = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = tokenizer.encode(prompt, return_tensors="pt")
        except:
            # 回退方案
            prompt = f"System: {system_message}\nUser: {user_input}\nAssistant:"
            inputs = tokenizer.encode(prompt, return_tensors="pt")
    else:
        # 对于不支持chat_template的模型
        prompt = f"System: {system_message}\nUser: {user_input}\nAssistant:"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # 提取回复
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        except:
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 尝试从响应中提取助手部分
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
    else:
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):] if len(full_response) > len(prompt) else full_response
    
    return response.strip()


def get_cache_key(model_name: str, test_samples: List[Dict]) -> str:
    """生成缓存键"""
    content = model_name + "".join([json.dumps(s, sort_keys=True) for s in test_samples])
    return hashlib.md5(content.encode()).hexdigest()


def get_or_generate_predictions(model_config: Dict, test_samples: List[Dict]) -> List[str]:
    """获取或生成模型预测结果"""
    model_name = model_config['display_name']
    cache_key = get_cache_key(model_name, test_samples)
    cache_file = os.path.join(args.cache_dir, f"{cache_key}.json")
    
    # 检查缓存
    if not args.force_regenerate and os.path.exists(cache_file):
        print(f"  从缓存加载 {model_name} 的预测结果...")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['predictions']
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # 生成预测
    print(f"  生成 {model_name} 的回复...")
    predictions = []
    
    for sample in tqdm(test_samples, desc=f"生成 {model_name} 的回复"):
        user_input = sample.get("input", "")
        
        try:
            prediction = generate_response(
                model, 
                tokenizer, 
                user_input, 
                max_length=args.max_length
            )
            predictions.append(prediction)
        except Exception as e:
            print(f"  生成回复时出错: {e}")
            predictions.append("")
    
    # 保存到缓存
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name,
            'predictions': predictions,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    
    # 卸载模型
    unload_model(model, tokenizer)
    
    return predictions


def evaluate_single_model(model_config: Dict, test_samples: List[Dict], evaluator: MentalHealthEvaluator) -> Dict:
    """评估单个模型"""
    model_name = model_config['display_name']
    print(f"\n评估模型: {model_name}")
    print("=" * 60)
    
    # 获取预测结果
    predictions = get_or_generate_predictions(model_config, test_samples)
    
    # 准备参考数据和用户输入
    references = [sample.get("output", "") for sample in test_samples]
    user_inputs = [sample.get("input", "") for sample in test_samples]
    
    # 计算指标
    metrics = {}
    
    if len(predictions) > 0 and len(references) > 0:
        # 文本质量指标
        text_metrics = evaluator.compute_bleu_rouge(predictions, references)
        metrics.update(text_metrics)
        
        # 语义相似度
        semantic_metrics = evaluator.compute_semantic_similarity(predictions, references)
        metrics.update(semantic_metrics)
        
        # 心理健康特定指标
        metrics['safety'] = evaluator.compute_safety_scores(predictions)
        metrics['empathy'] = evaluator.compute_empathy_indicators(predictions)
        metrics['relevance'] = evaluator.compute_relevance_score(predictions, user_inputs)
        
        # 计算回复长度
        response_lengths = [len(pred) for pred in predictions]
        metrics['avg_response_length'] = np.mean(response_lengths)
        metrics['std_response_length'] = np.std(response_lengths)
        
        # 综合分数
        metrics['overall_score'] = (
            metrics['bleu'] * 0.15 +
            metrics['rougeL'] * 0.15 +
            metrics['tfidf_cosine'] * 0.15 +
            metrics['safety'] * 0.2 +
            metrics['empathy'] * 0.2 +
            metrics['relevance'] * 0.15
        )
    
    print(f"评估完成: {model_name}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    return metrics, predictions


def get_model_configs() -> List[Dict]:
    """获取要评估的模型配置列表"""
    model_configs = []
    
    # 1. 基础模型（Qwen）
    model_configs.append({
        'model_name': args.base_model,
        'display_name': f"Base {args.base_model.split('/')[-1]}",
        'adapter_path': None,
        'use_peft': False
    })
    
    # 评估基础模型的微调版本
    if os.path.exists(args.adapter_path):
        model_configs.append({
            'model_name': args.base_model,
            'display_name': f"Finetuned {args.base_model.split('/')[-1]}",
            'adapter_path': args.adapter_path,
            'use_peft': True
        })
    elif args.adapter_path:
        print(f"Warning: Adapter path for base model does not exist: {args.adapter_path}")
    
    # 2. ChatGLM
    if args.eval_chatglm:
        # ChatGLM基础版本
        model_configs.append({
            'model_name': args.chatglm_model,
            'display_name': f"Base {args.chatglm_model.split('/')[-1]}",
            'adapter_path': None,
            'use_peft': False
        })
        
        # ChatGLM微调版本
        if args.chatglm_adapter_path and os.path.exists(args.chatglm_adapter_path):
            model_configs.append({
                'model_name': args.chatglm_model,
                'display_name': f"Finetuned {args.chatglm_model.split('/')[-1]}",
                'adapter_path': args.chatglm_adapter_path,
                'use_peft': True
            })
        elif args.chatglm_adapter_path:
            print(f"Warning: ChatGLM adapter path does not exist: {args.chatglm_adapter_path}")
    
    # 3. Qwen2.5
    if args.eval_qwen2_5:
        # Qwen2.5基础版本
        model_configs.append({
            'model_name': args.qwen2_5_model,
            'display_name': f"Base {args.qwen2_5_model.split('/')[-1]}",
            'adapter_path': None,
            'use_peft': False
        })
        
        # Qwen2.5微调版本
        if args.qwen2_5_adapter_path and os.path.exists(args.qwen2_5_adapter_path):
            model_configs.append({
                'model_name': args.qwen2_5_model,
                'display_name': f"Finetuned {args.qwen2_5_model.split('/')[-1]}",
                'adapter_path': args.qwen2_5_adapter_path,
                'use_peft': True
            })
        elif args.qwen2_5_adapter_path:
            print(f"Warning: Qwen2.5 adapter path does not exist: {args.qwen2_5_adapter_path}")
    
    return model_configs


def get_test_samples(dataset, num_samples: int) -> List[Dict]:
    """获取测试样本"""
    # 设置随机种子以确保一致性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 准备测试集
    if "test" in dataset:
        test_dataset = dataset["test"]
    elif "validation" in dataset:
        test_dataset = dataset["validation"]
    else:
        # 从训练集划分
        train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        test_dataset = train_test_split["test"]
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 选择样本
    if len(test_dataset) > num_samples:
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        test_samples = [test_dataset[int(i)] for i in indices]
    else:
        test_samples = [test_dataset[i] for i in range(len(test_dataset))]
    
    return test_samples


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """创建对比表格"""
    metrics = ['bleu', 'rougeL', 'tfidf_cosine', 'safety', 'empathy', 'relevance', 'overall_score']
    
    df_data = {}
    for model_name, metrics_dict in results.items():
        row = []
        for metric in metrics:
            if metric in metrics_dict:
                row.append(metrics_dict[metric])
            else:
                row.append(0.0)
        df_data[model_name] = row
    
    df = pd.DataFrame(df_data, index=metrics)
    return df.T


def plot_comparison(results: Dict[str, Dict], save_path: str = "model_comparison.png"):
    """绘制对比图表"""
    metrics_to_plot = ['bleu', 'rougeL', 'safety', 'empathy', 'relevance', 'overall_score']
    
    plt.figure(figsize=(15, 10))
    
    # 设置样式
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(results))
    
    # 创建子图
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        
        model_names = []
        scores = []
        
        for model_name, metrics in results.items():
            if metric in metrics:
                model_names.append(model_name)
                scores.append(metrics[metric])
        
        bars = plt.bar(model_names, scores, color=colors[:len(model_names)])
        plt.title(f'{metric.upper()} Score', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # 在柱子上添加数值
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def qualitative_analysis(predictions_dict: Dict[str, List[str]], references: List[str], 
                         user_inputs: List[str], num_examples: int = 5):
    """定性分析 - 展示具体例子"""
    print("\n" + "="*80)
    print("定性分析示例")
    print("="*80)
    
    model_names = list(predictions_dict.keys())
    
    for i in range(min(num_examples, len(references))):
        print(f"\n--- 示例 {i+1} ---")
        print(f"👤 用户输入: {user_inputs[i][:200]}...")
        print(f"📝 参考回复: {references[i][:200]}...")
        print("-" * 60)
        
        for model_name in model_names:
            if i < len(predictions_dict[model_name]):
                pred = predictions_dict[model_name][i]
                print(f"🤖 {model_name}:")
                print(f"   {pred[:200]}..." if len(pred) > 200 else f"   {pred}")
        print("=" * 80)


def main():
    """主函数"""
    print("="*80)
    print("心理健康对话模型评估系统")
    print("="*80)
    
    # 1. 加载测试数据
    print("加载测试数据...")
    from datasets import load_dataset
    dataset = load_dataset("ShenLab/MentalChat16K")
    
    # 2. 获取测试样本
    test_samples = get_test_samples(dataset, args.num_samples)
    print(f"使用 {len(test_samples)} 个测试样本")
    
    # 3. 获取模型配置
    model_configs = get_model_configs()
    print(f"将评估 {len(model_configs)} 个模型:")
    for config in model_configs:
        print(f"  - {config['display_name']}")
    
    # 4. 初始化评估器
    evaluator = MentalHealthEvaluator()
    
    # 5. 逐个评估模型
    all_results = {}
    all_predictions = {}
    
    for model_config in model_configs:
        metrics, predictions = evaluate_single_model(model_config, test_samples, evaluator)
        all_results[model_config['display_name']] = metrics
        all_predictions[model_config['display_name']] = predictions
    
    # 6. 生成对比结果
    comparison_df = create_comparison_table(all_results)
    
    print("\n" + "="*80)
    print("模型对比结果")
    print("="*80)
    print(comparison_df.round(4))
    
    # 7. 保存结果
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{args.save_dir}/model_comparison_results_{timestamp}.csv"
    comparison_df.to_csv(results_file)
    print(f"结果已保存到: {results_file}")
    
    # 8. 保存详细结果
    detailed_file = f"{args.save_dir}/detailed_comparison_results_{timestamp}.json"
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump({
            'results': all_results,
            'predictions': all_predictions,
            'references': [s.get("output", "") for s in test_samples],
            'user_inputs': [s.get("input", "") for s in test_samples],
            'config': vars(args)
        }, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {detailed_file}")
    
    # 9. 生成可视化
    plot_file = f"{args.save_dir}/model_comparison_{timestamp}.png"
    plot_comparison(all_results, plot_file)
    
    # 10. 定性分析
    references = [s.get("output", "") for s in test_samples]
    user_inputs = [s.get("input", "") for s in test_samples]
    qualitative_analysis(all_predictions, references, user_inputs)
    
    # 11. 总结
    print("\n" + "="*80)
    print("评估总结")
    print("="*80)
    
    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1].get('overall_score', 0))
        print(f"🏆 最佳模型: {best_model[0]} (综合分数: {best_model[1]['overall_score']:.4f})")
        
        # 显示各项最佳
        metrics = ['bleu', 'rougeL', 'safety', 'empathy', 'relevance']
        for metric in metrics:
            best_for_metric = max(all_results.items(), key=lambda x: x[1].get(metric, 0))
            print(f"📊 最佳 {metric}: {best_for_metric[0]} ({best_for_metric[1][metric]:.4f})")
    
    print(f"\n评估完成! 所有模型已从GPU中移除。")
    print(f"结果文件: {results_file}")
    print(f"图表文件: {plot_file}")
    
    return all_results, comparison_df


if __name__ == "__main__":
    results, df = main()