import sys
import os
import argparse
import re
from lite_lcel import PromptTemplate, MockLLM, StringOutputParser, ConsoleCallbackHandler

# ANSI kaçış kodları ile renklendirme
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title: str):
    print(f"\n{BOLD}{BLUE}======================================================================")
    print(f" {title}")
    print(f"======================================================================{RESET}")

def run_playground():
    print_header("✨ LITE-LCEL REPL PLAYGROUND (TRACE & CALLBACKS) ✨")
    print("Bu ortamda basit bir Prompt | LLM | Parser zincirini test edebilirsiniz.")
    print("LLM olarak kural tabanlı 'MockLLM' kullanılacaktır.")
    
    template_str = input(f"\n{BOLD}{YELLOW}Prompt Şablonunu girin (Varsayılan: 'Bana {{konu}} hakkında bilgi ver'): {RESET}").strip()
    if not template_str:
        template_str = "Bana {konu} hakkında bilgi ver"
        
    prompt = PromptTemplate(template=template_str)
    model = MockLLM(model_name="Playground-GPT-4", delay=0.2)
    parser = StringOutputParser()
    
    chain = prompt | model | parser
    
    # Şablondaki yer tutucuları çıkar
    placeholders = re.findall(r"\{([^}]+)\}", template_str)
    
    inputs = {}
    if placeholders:
        print(f"\n{BOLD}{CYAN}Lütfen şablondaki değişkenleri doldurun:{RESET}")
        for ph in placeholders:
            val = input(f"  {ph}: ").strip()
            inputs[ph] = val
    else:
        user_input = input(f"\n{BOLD}{CYAN}Girdi metnini yazın: {RESET}").strip()
        inputs = {"input": user_input}
        
    print(f"\n{BOLD}{GREEN}⚡ Zincir Çalıştırılıyor (ConsoleCallbackHandler izleyicisi aktif)...{RESET}")
    config = {"callbacks": [ConsoleCallbackHandler()]}
    
    try:
        result = chain.invoke(inputs, config=config)
        print(f"\n{BOLD}{MAGENTA}🎉 Zincir Yanıtı:{RESET}\n{result}")
    except Exception as e:
        print(f"\n{BOLD}{RED}❌ Hata oluştu: {e}{RESET}")

def run_example(example_name: str):
    print_header(f"🚀 {example_name.upper()} ÖRNEĞİ ÇALIŞTIRILIYOR 🚀")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_path = os.path.join(base_dir, "examples", f"{example_name}.py")
    
    if not os.path.exists(example_path):
        print(f"{RED}Hata: Örnek bulunamadı: {example_path}{RESET}")
        return
        
    import subprocess
    # Python yorumlayıcısı ile örneği çalıştır
    subprocess.run([sys.executable, example_path])

def main():
    parser = argparse.ArgumentParser(description="Lite-LCEL CLI - Hafif LCEL Kütüphanesi Playground ve Örnek Çalıştırıcı")
    parser.add_argument("--playground", action="store_true", help="İnteraktif REPL Playground başlat")
    parser.add_argument("--example", choices=[
        "basic_chain", "async_chain", "streaming_chain", "parallel_chain",
        "branch_and_fallback", "stateful_chain", "tool_calling_chain",
        "autonomous_agent", "retry_chain", "rag_chain"
    ], help="Belirtilen örnek senaryoyu çalıştır")
    
    args = parser.parse_args()
    
    if args.playground:
        run_playground()
        return
        
    if args.example:
        run_example(args.example)
        return
        
    # Herhangi bir argüman geçilmediyse interaktif menüyü göster
    while True:
        print_header("🤖 LITE-LCEL İNTERAKTİF YÖNETİM PANELİ 🤖")
        print("1. İnteraktif LCEL REPL Playground Başlat")
        print("2. Örnek Senaryolardan Birini Çalıştır")
        print("3. Çıkış")
        
        choice = input(f"\n{BOLD}Seçiminiz (1-3): {RESET}").strip()
        
        if choice == "1":
            run_playground()
        elif choice == "2":
            print(f"\n{BOLD}{CYAN}Kullanılabilir Örnekler:{RESET}")
            examples = [
                "basic_chain", "async_chain", "streaming_chain", "parallel_chain",
                "branch_and_fallback", "stateful_chain", "tool_calling_chain",
                "autonomous_agent", "retry_chain", "rag_chain"
            ]
            for idx, ex in enumerate(examples, 1):
                print(f"  {idx}. {ex}")
            
            ex_choice = input(f"\n{BOLD}Çalıştırmak istediğiniz örneğin numarası: {RESET}").strip()
            try:
                ex_idx = int(ex_choice) - 1
                if 0 <= ex_idx < len(examples):
                    run_example(examples[ex_idx])
                else:
                    print(f"{RED}Geçersiz numara!{RESET}")
            except ValueError:
                print(f"{RED}Lütfen bir sayı girin!{RESET}")
        elif choice == "3":
            print(f"\n{GREEN}Güle güle! 👋{RESET}\n")
            break
        else:
            print(f"{RED}Geçersiz seçim!{RESET}")

if __name__ == "__main__":
    main()
