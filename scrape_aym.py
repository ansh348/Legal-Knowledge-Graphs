#!/usr/bin/env python3
"""
scrape_aym.py  (Playwright version)

Scrape bireysel başvuru decisions from AYM Kararlar Bilgi Bankası.
The site is a JS SPA, so we use Playwright to render pages.

Setup
-----
    pip install playwright
    playwright install chromium

Usage
-----
    # Scrape 10 ihlal decisions
    python scrape_aym.py --n 10 --output_dir aym_cases

    # Then run extraction:
    python run_iltur.py --dataset aym --local_dir aym_cases --n 10 --version v4 --concurrent 5
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Error: playwright not installed.")
    print("  pip install playwright")
    print("  playwright install chromium")
    sys.exit(1)


# Known bireysel başvuru numbers (ihlal decisions with rich reasoning).
# Sourced from AYM basın duyuruları and search index.
KNOWN_CASES = [
    "2016/25923",   # Remziye Duman
    "2019/6266",    # Mehmet Özcan
    "2019/13338",   # Ümran Özkan (GK)
    "2021/53760",   # Erol Eski
    "2018/20423",   # Bahri Alataş
    "2019/35433",   # Zekeriya Acar
    "2018/24998",   # Ali Gönültaş
    "2022/11809",   # Halil Sarıkaya
    "2020/20382",   # Ferihan Beyoğlu
    "2020/15944",   # Cem Özberk
    "2019/42188",   # Betül Özbey Bayındır
    "2020/5754",    # Musa Özalp
    "2021/20360",   # Menduh Ataç
    "2020/2697",    # FYA Turizm
    "2022/19376",   # Sait Görmüş
]

BASE_URL = "https://kararlarbilgibankasi.anayasa.gov.tr/BB"


def _clean_text(raw: str) -> str:
    """Clean extracted text: strip nav/footer, normalize whitespace."""
    # Find the start of actual content
    markers = [
        r'T\.?\s*C\.?\s*ANAYASA\s*MAHKEMESİ',
        r'TÜRKİYE\s+CUMHURİYETİ\s+ANAYASA\s+MAHKEMESİ',
        r'Başvuru\s+Numarası\s*:',
        r'Başvuru\s+No\s*:',
        r'B\.\s*No\s*:',
        r'I\.\s+BAŞVURU',
        r'I\.\s+OLAYLAR',
    ]

    start_idx = 0
    for marker in markers:
        m = re.search(marker, raw)
        if m:
            start_idx = m.start()
            break

    # Find end of content (before footer)
    end_markers = [
        r'ADRES\s*Ahlatlıbel',
        r'İLETİŞİM\s+BİLGİLERİ',
        r'Telefon\s+312\s+463',
    ]

    end_idx = len(raw)
    for marker in end_markers:
        m = re.search(marker, raw[start_idx:])
        if m:
            end_idx = start_idx + m.start()
            break

    text = raw[start_idx:end_idx].strip()

    # Normalize: keep paragraph breaks, collapse extra whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def _make_filename(basvuru_no: str) -> str:
    """Convert '2019/6266' to 'aym_2019_6266'."""
    return "aym_" + re.sub(r'[^0-9A-Za-z]+', '_', basvuru_no).strip('_')


async def scrape_decision(page, basvuru_no: str, output_dir: Path) -> bool:
    """Scrape a single AYM decision using Playwright."""
    filename = _make_filename(basvuru_no)
    out_file = output_dir / f"{filename}.txt"

    if out_file.exists() and out_file.stat().st_size > 500:
        print(f"  {basvuru_no} — already exists ({out_file.stat().st_size:,} bytes), skipping")
        return True

    url = f"{BASE_URL}/{basvuru_no}"
    print(f"  Fetching {basvuru_no} from {url} ...")

    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        # Wait for dynamic content to render
        await page.wait_for_timeout(3000)

        # Try specific content selectors first
        content = None
        selectors = [
            "#kararMetni", ".karar-metni", ".karar-icerik",
            ".content-area", "#icerik", ".icerik",
            "article", ".karar", "#karar", "main",
        ]

        for sel in selectors:
            try:
                elem = await page.query_selector(sel)
                if elem:
                    content = await elem.inner_text()
                    if content and len(content) > 500:
                        break
            except Exception:
                continue

        # Fallback: get all body text
        if not content or len(content) < 500:
            content = await page.inner_text("body")

        if not content or len(content) < 200:
            print(f"    ⚠ No content found for {basvuru_no}")
            return False

        text = _clean_text(content)

        if len(text) < 200:
            print(f"    ⚠ Content too short after cleaning ({len(text)} chars)")
            return False

        out_file.write_text(text, encoding="utf-8")
        print(f"    ✅ Saved ({len(text):,} chars)")
        return True

    except Exception as e:
        print(f"    ❌ Error: {type(e).__name__}: {e}")
        return False


async def scrape_search_results(page, n: int) -> list:
    """Try to gather başvuru numbers from the search results page."""
    print("\nGathering case numbers from search page...")
    search_url = "https://kararlarbilgibankasi.anayasa.gov.tr/Ara?Sonuc%5B0%5D=5"

    try:
        await page.goto(search_url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(3000)

        links = await page.query_selector_all('a[href*="/BB/"]')
        case_numbers = []
        for link in links:
            href = await link.get_attribute("href")
            if href:
                m = re.search(r'/BB/(\d{4}/\d+)', href)
                if m and m.group(1) not in case_numbers:
                    case_numbers.append(m.group(1))

        if case_numbers:
            print(f"  Found {len(case_numbers)} case numbers from search")
            return case_numbers[:n]

    except Exception as e:
        print(f"  Search scraping failed: {e}")

    return []


async def main_async(n: int, output_dir: str):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Scraping {n} AYM bireysel başvuru decisions...")
    print(f"Output: {out_dir}/\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="tr-TR",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        # First try to get case numbers from the search page
        dynamic_cases = await scrape_search_results(page, n)

        # Merge with known cases, preferring dynamic results
        case_numbers = dynamic_cases[:n]
        if len(case_numbers) < n:
            for kc in KNOWN_CASES:
                if kc not in case_numbers:
                    case_numbers.append(kc)
                if len(case_numbers) >= n:
                    break
        case_numbers = case_numbers[:n]

        print(f"\nWill scrape {len(case_numbers)} decisions:")
        for cn in case_numbers:
            print(f"  - {cn}")
        print()

        saved = 0
        for idx, cn in enumerate(case_numbers):
            print(f"[{idx+1}/{len(case_numbers)}]", end="")
            ok = await scrape_decision(page, cn, out_dir)
            if ok:
                saved += 1
            await asyncio.sleep(2)  # polite delay

        await browser.close()

    print(f"\n{'='*50}")
    print(f"Done. Saved {saved}/{len(case_numbers)} decisions to {out_dir}/")
    if saved > 0:
        print(f"\nNext step:")
        print(f"  python run_iltur.py --dataset aym --local_dir {out_dir} --n {saved} --version v4 --concurrent 5")
    else:
        print("\nNo decisions scraped. Try manual download:")
        print("  1. Go to https://kararlarbilgibankasi.anayasa.gov.tr")
        print("  2. Search for Esas (İhlal) decisions")
        print("  3. Copy full text → save as .txt files")
        print(f"  4. Put files in {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Scrape AYM decisions via Playwright")
    parser.add_argument("--n", type=int, default=10, help="Number of decisions to scrape")
    parser.add_argument("--output_dir", type=str, default="aym_cases", help="Output directory")
    args = parser.parse_args()
    asyncio.run(main_async(args.n, args.output_dir))


if __name__ == "__main__":
    main()