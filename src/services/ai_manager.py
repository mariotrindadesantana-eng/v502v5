#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQV30 Enhanced v2.0 - AI Manager com Sistema de Fallback
Gerenciador inteligente de múltiplas IAs com fallback automático
"""

import os
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import google.generativeai as genai
import openai
import requests

logger = logging.getLogger(__name__)

class AIManager:
    """Gerenciador de IAs com sistema de fallback automático"""

    def __init__(self):
        """Inicializa o gerenciador de IAs"""
        self.providers = {
            'gemini': {
                'client': None,
                'available': False,
                'priority': 1,
                'rate_limit_reset': None,
                'error_count': 0,
                'model': 'gemini-1.5-flash'  # Modelo mais eficiente
            },
            'openai': {
                'client': None,
                'available': False,
                'priority': 2,
                'rate_limit_reset': None,
                'error_count': 0,
                'model': 'gpt-3.5-turbo'  # Modelo mais estável
            },
            'huggingface': {
                'client': None,
                'available': False,
                'priority': 3,
                'rate_limit_reset': None,
                'error_count': 0,
                'models': [
                    "microsoft/DialoGPT-medium",
                    "microsoft/DialoGPT-large",
                    "facebook/blenderbot-400M-distill",
                    "google/flan-t5-base",
                    "HuggingFaceH4/zephyr-7b-beta"
                ],
                'current_model_index': 0
            }
        }

        self.initialize_providers()
        logger.info(f"AI Manager inicializado com {len([p for p in self.providers.values() if p['available']])} provedores disponíveis")

    def initialize_providers(self):
        """Inicializa todos os provedores de IA"""

        # Inicializa Gemini
        try:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.providers['gemini']['client'] = genai.GenerativeModel("gemini-1.5-flash")
                self.providers['gemini']['available'] = True
                logger.info("✅ Gemini Flash inicializado com sucesso")
        except Exception as e:
            logger.warning(f"⚠️ Falha ao inicializar Gemini: {str(e)}")

        # Inicializa OpenAI
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                openai.api_key = openai_key
                self.providers["openai"]["client"] = openai.OpenAI(api_key=openai_key)
                self.providers["openai"]["available"] = True
                logger.info("✅ OpenAI inicializado com sucesso")
        except Exception as e:
            logger.warning(f"⚠️ Falha ao inicializar OpenAI: {str(e)}")

        # Inicializa HuggingFace
        try:
            hf_key = os.getenv('HUGGINGFACE_API_KEY')
            if hf_key:
                self.providers['huggingface']['client'] = {
                    'api_key': hf_key,
                    'base_url': 'https://api-inference.huggingface.co/models/'
                }
                self.providers['huggingface']['available'] = True
                logger.info("✅ HuggingFace inicializado com sucesso")
        except Exception as e:
            logger.warning(f"⚠️ Falha ao inicializar HuggingFace: {str(e)}")

    def get_best_provider(self) -> Optional[str]:
        """Retorna o melhor provedor disponível"""
        available_providers = [
            (name, provider) for name, provider in self.providers.items() 
            if provider['available'] and provider['error_count'] < 5
        ]

        if not available_providers:
            # Reset error counts se todos falharam
            for provider in self.providers.values():
                provider['error_count'] = 0
            available_providers = [
                (name, provider) for name, provider in self.providers.items() 
                if provider['available']
            ]

        if available_providers:
            # Ordena por prioridade e menor número de erros
            available_providers.sort(key=lambda x: (x[1]['priority'], x[1]['error_count']))
            return available_providers[0][0]

        return None

    def generate_analysis(self, prompt: str, max_tokens: int = 8192) -> Optional[str]:
        """Gera análise usando o melhor provedor disponível"""

        provider_name = self.get_best_provider()
        if not provider_name:
            logger.error("❌ Nenhum provedor de IA disponível")
            raise Exception("NENHUM PROVEDOR DE IA DISPONÍVEL: Configure pelo menos uma API de IA")

        logger.info(f"🤖 Usando provedor: {provider_name}")

        try:
            if provider_name == 'gemini':
                return self._generate_with_gemini(prompt, max_tokens)
            elif provider_name == 'openai':
                return self._generate_with_openai(prompt, max_tokens)
            elif provider_name == 'huggingface':
                return self._generate_with_huggingface(prompt, max_tokens)
        except Exception as e:
            logger.error(f"❌ Erro no provedor {provider_name}: {str(e)}")
            self.providers[provider_name]['error_count'] += 1

            # Tenta próximo provedor
            fallback_result = self._try_fallback(prompt, max_tokens, exclude=[provider_name])
            if not fallback_result:
                raise Exception(f"TODOS OS PROVEDORES DE IA FALHARAM: Último erro - {str(e)}")
            return fallback_result

        raise Exception("ERRO INESPERADO: Nenhum provedor processou a requisição")

    def _generate_with_gemini(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Gera conteúdo usando Gemini"""
        try:
            client = self.providers['gemini']['client']

            generation_config = {
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 64,
                'max_output_tokens': min(max_tokens, 2048),  # Reduz para evitar quota
                'candidate_count': 1
            }

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            response = client.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            if response.text:
                logger.info(f"✅ Gemini gerou {len(response.text)} caracteres")
                return response.text
            else:
                raise Exception("Resposta vazia do Gemini")

        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                logger.warning(f"⚠️ Gemini atingiu limite de quota: {str(e)}")
                self.providers['gemini']['rate_limit_reset'] = time.time() + 3600  # 1 hora
            raise e

    def _generate_with_openai(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Gera conteúdo usando OpenAI"""
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not found")
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise de mercado ultra-detalhada."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=min(max_tokens, 1500),
                temperature=0.7,
                top_p=0.95
            )

            content = response.choices[0].message.content
            if content:
                logger.info(f"✅ OpenAI gerou {len(content)} caracteres")
                return content
            else:
                raise Exception("Resposta vazia do OpenAI")

        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                logger.warning(f"⚠️ OpenAI atingiu limite de quota: {str(e)}")
                self.providers['openai']['rate_limit_reset'] = time.time() + 3600
            raise e

    def _generate_with_huggingface(self, prompt: str, max_tokens: int) -> Optional[str]:
        """Gera conteúdo usando HuggingFace com rotação de modelos"""
        hf_config = self.providers['huggingface']
        models = hf_config['models']

        # Tenta todos os modelos disponíveis
        for attempt in range(len(models)):
            current_model = models[hf_config['current_model_index']]

            try:
                url = f"{hf_config['client']['base_url']}{current_model}"
                headers = {
                    "Authorization": f"Bearer {hf_config['client']['api_key']}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0.9,
                        "return_full_text": False,
                        "do_sample": True,
                        "top_p": 0.95
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False
                    }
                }

                response = requests.post(url, headers=headers, json=payload, timeout=60)

                if response.status_code == 200:
                    data = response.json()

                    if isinstance(data, list) and len(data) > 0:
                        if "generated_text" in data[0]:
                            content = data[0]["generated_text"]
                        elif "text" in data[0]:
                            content = data[0]["text"]
                        else:
                            content = str(data[0])

                        # Remove prompt se incluído
                        if content.startswith(prompt):
                            content = content[len(prompt):].strip()

                        if content:
                            logger.info(f"✅ HuggingFace ({current_model}) gerou {len(content)} caracteres")
                            return content

                elif response.status_code == 503:
                    logger.warning(f"⚠️ Modelo {current_model} carregando, tentando próximo...")
                    # Rotaciona para próximo modelo
                    hf_config['current_model_index'] = (hf_config['current_model_index'] + 1) % len(models)
                    continue
                else:
                    logger.warning(f"⚠️ Erro {response.status_code} no modelo {current_model}")
                    hf_config['current_model_index'] = (hf_config['current_model_index'] + 1) % len(models)
                    continue

            except Exception as e:
                logger.warning(f"⚠️ Erro no modelo {current_model}: {str(e)}")
                hf_config['current_model_index'] = (hf_config['current_model_index'] + 1) % len(models)
                continue

        raise Exception("Todos os modelos HuggingFace falharam")

    def _try_fallback(self, prompt: str, max_tokens: int, exclude: List[str] = None) -> Optional[str]:
        """Tenta usar provedor de fallback"""
        exclude = exclude or []

        for provider_name in ['gemini', 'openai', 'huggingface']:
            if provider_name in exclude:
                continue

            if not self.providers[provider_name]['available']:
                continue

            if self.providers[provider_name]['error_count'] >= 5:
                continue

            logger.info(f"🔄 Tentando fallback para: {provider_name}")

            try:
                if provider_name == 'gemini':
                    return self._generate_with_gemini(prompt, max_tokens)
                elif provider_name == 'openai':
                    return self._generate_with_openai(prompt, max_tokens)
                elif provider_name == 'huggingface':
                    return self._generate_with_huggingface(prompt, max_tokens)
            except Exception as e:
                logger.warning(f"⚠️ Fallback {provider_name} falhou: {str(e)}")
                self.providers[provider_name]['error_count'] += 1
                continue

        logger.error("❌ Todos os provedores de fallback falharam")
        return None

    def get_provider_status(self) -> Dict[str, Any]:
        """Retorna status de todos os provedores"""
        status = {}

        for name, provider in self.providers.items():
            status[name] = {
                'available': provider['available'],
                'priority': provider['priority'],
                'error_count': provider['error_count'],
                'rate_limited': (provider.get('rate_limit_reset') or 0) > time.time()
            }

            if name == 'huggingface' and provider['available']:
                status[name]['current_model'] = provider['models'][provider['current_model_index']]
                status[name]['available_models'] = len(provider['models'])

        return status

    def reset_provider_errors(self, provider_name: str = None):
        """Reset contadores de erro"""
        if provider_name:
            if provider_name in self.providers:
                self.providers[provider_name]['error_count'] = 0
                logger.info(f"🔄 Reset erros do provedor: {provider_name}")
        else:
            for provider in self.providers.values():
                provider['error_count'] = 0
            logger.info("🔄 Reset erros de todos os provedores")

    def _clean_ai_response(self, response: str) -> str:
        """Limpa resposta da IA para extrair JSON puro"""
        if not response:
            return response

        # Remove markdown de código JSON
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.rfind("```")
            if end > start:
                return response[start:end].strip()

        # Remove qualquer markdown de código
        elif "```" in response:
            start = response.find("```") + 3
            end = response.rfind("```")
            if end > start:
                return response[start:end].strip()

        # Remove explicações antes do JSON
        lines = response.split('\n')
        json_started = False
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith('{') or json_started:
                json_started = True
                cleaned_lines.append(line)
            elif line.startswith('}'):
                cleaned_lines.append(line)
                break

        if cleaned_lines:
            return '\n'.join(cleaned_lines)

        return response.strip()

# Instância global
ai_manager = AIManager()