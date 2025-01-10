import os
from typing import List, Optional, Callable, Any
import aiohttp
import asyncio
from tqdm import tqdm
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class OpenAITranslator:
    def __init__(self, api_key: str = None, api_base: str = "https://api.openai.com/v1"):
        """Initialize OpenAI translator
        
        Args:
            api_key: OpenAI API key, if None, get from environment variable OPENAI_API_KEY
            api_base: API base URL, can be set to mirror site address
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.api_base = api_base.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = aiohttp.ClientTimeout(total=30)  # Set timeout to 30 seconds

    async def translate_text(self, text: str, target_language: str = "English", 
                           session: Optional[aiohttp.ClientSession] = None,
                           max_retries: int = 3,
                           retry_delay: float = 2.0) -> str:
        """Asynchronous translation of a single text
        
        Args:
            text: Text to translate
            target_language: Target language, default is English
            session: aiohttp session, if None, create a new session
            max_retries: Maximum number of retries
            retry_delay: Retry delay (seconds)
        
        Returns:
            Translated text
        """
        retry_count = 0
        start_time = time.time()
        
        while retry_count <= max_retries:
            try:
                url = f"{self.api_base}/chat/completions"
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": f"You are a professional translator, please translate the following text into {target_language}, keep the original tone and style, only return the translation result."},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.3
                }
                
                should_close = session is None
                if session is None:
                    session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout)
                
                try:
                    async with session.post(url, json=payload) as response:
                        response.raise_for_status()
                        result = await response.json()
                        
                        if "error" in result:
                            raise Exception(result["error"]["message"])
                        
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 10:  # record request if it takes too long
                            logging.warning(f"""
Request took too long ({elapsed_time:.2f} seconds):
Original text: {text[:100]}...
Target language: {target_language}
Retry count: {retry_count}
Response: {result}
""")
                        
                        return result["choices"][0]["message"]["content"].strip()
                finally:
                    if should_close:
                        await session.close()
                
            except asyncio.TimeoutError:
                retry_count += 1
                if retry_count <= max_retries:
                    logging.warning(f"Timeout, retrying {retry_count} times... (Text: {text[:50]}...)")
                    await asyncio.sleep(retry_delay * retry_count)  # 指数退避
                else:
                    logging.error(f"Request timed out, returning original text. (Text: {text[:50]}...)")
                    return text
                    
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    logging.warning(f"Error: {str(e)}, retrying {retry_count} times... (Text: {text[:50]}...)")
                    await asyncio.sleep(retry_delay * retry_count)
                else:
                    logging.error(f"Failed after {max_retries} retries: {str(e)}, returning original text. (Text: {text[:50]}...)")
                    return text

    async def translate_batch(self, texts: List[str], target_language: str = "English", max_concurrency: int = 5) -> List[str]:
        """Asynchronous batch translation of text
        
        Args:
            texts: List of text to translate
            target_language: Target language, default is English
            max_concurrency: Maximum number of concurrent requests, default is 5
        
        Returns:
            List of translated text
        """
        async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
            semaphore = asyncio.Semaphore(max_concurrency)
            pbar = tqdm(total=len(texts), desc="Translation progress")
            
            async def translate_with_semaphore(text: str, index: int) -> tuple[int, str]:
                async with semaphore:
                    result = await self.translate_text(text, target_language, session)
                    pbar.update(1)
                    return index, result
            
            # Create task list, including index information
            tasks = [translate_with_semaphore(text, i) for i, text in enumerate(texts)]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()
            
            # Process results, including possible exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Task {i} failed: {str(result)}")
                    processed_results.append((i, texts[i]))  # Return original text if exception occurs
                else:
                    processed_results.append(result)
            
            # Sort results by index
            sorted_results = sorted(processed_results, key=lambda x: x[0])
            return [result[1] for result in sorted_results]
