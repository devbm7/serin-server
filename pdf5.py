import os
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

# ReportLab for PDF generation
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
	SimpleDocTemplate,
	Paragraph,
	Spacer,
	Table,
	TableStyle,
	Image,
)
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader


def _load_env() -> None:
	"""Load environment variables from .env files in common locations if present."""
	try:
		# Always try current working directory first
		load_dotenv()

		# Also try the repo root and known subdirs
		module_dir = os.path.dirname(__file__)
		root_dir = os.path.dirname(module_dir)

		# Root .env
		root_env = os.path.join(root_dir, ".env")
		if os.path.isfile(root_env):
			load_dotenv(root_env)

		# Job Transformer .env
		jt_env = os.path.join(root_dir, "Job Transformer", ".env")
		if os.path.isfile(jt_env):
			load_dotenv(jt_env)
	except Exception:
		pass


def _init_supabase() -> Client:
	"""Initialize and return a Supabase client using env vars.

	Required env vars (pick first available):
	- SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL
	- SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY or NEXT_PUBLIC_SUPABASE_ANON_KEY
	"""
	supabase_url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
	supabase_key = (
		os.getenv("SUPABASE_SERVICE_ROLE_KEY")
		or os.getenv("SUPABASE_ANON_KEY")
		or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
	)
	if not supabase_url:
		raise RuntimeError("Missing SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) env var")
	if not supabase_key:
		raise RuntimeError(
			"Missing SUPABASE_SERVICE_ROLE_KEY (or a compatible anon key) env var"
		)
	return create_client(supabase_url, supabase_key)


	


	


 


	


def _find_header_image() -> Optional[str]:
	"""Try to find a header image in the repo to brand the PDF."""
	# Common root header image based on repo structure
	candidates = [
		os.path.join(os.path.dirname(os.path.dirname(__file__)), "Project-Polaris-Header.png"),
		os.path.join(os.path.dirname(__file__), "Project-Polaris-Header.png"),
	]
	for path in candidates:
		if os.path.isfile(path):
			return path
	return None


def _as_text(val: Any) -> str:
	if val is None:
		return "N/A"
	if isinstance(val, (dict, list)):
		try:
			return json.dumps(val, ensure_ascii=False, indent=2)
		except Exception:
			return str(val)
	return str(val)


def _para(text: Any, style: ParagraphStyle) -> Paragraph:
	"""Convert arbitrary value to a wrapped Paragraph with the given style."""
	return Paragraph(_as_text(text), style)


def _build_pdf_story(report: Dict[str, Any], meta: Dict[str, Any]):
	styles = getSampleStyleSheet()
	styles.add(
		ParagraphStyle(
			name="SectionHeading",
			parent=styles["Heading2"],
			spaceBefore=18,
			spaceAfter=8,
			textColor=colors.HexColor("#0A2540"),
			fontSize=16,
			leading=20,
		)
	)
	styles.add(
		ParagraphStyle(
			name="MetaLabel",
			parent=styles["Normal"],
			textColor=colors.HexColor("#555555"),
			fontSize=10,
			leading=14,
		)
	)
	styles.add(
		ParagraphStyle(
			name="BodyTextLarge",
			parent=styles["BodyText"],
			fontSize=11,
			leading=15,
		)
	)
	styles.add(
		ParagraphStyle(
			name="CellText",
			parent=styles["BodyText"],
			fontSize=10,
			leading=14,
			wordWrap='CJK',
		)
	)

	story = []

	# Optional header image
	header_img = _find_header_image()
	if header_img:
		try:
			# Scale header image to page width preserving aspect ratio
			reader = ImageReader(header_img)
			iw, ih = reader.getSize()
			# Target width approximates content width (LETTER minus margins)
			target_w = 6.8 * inch
			scale = target_w / float(iw)
			target_h = ih * scale
			im = Image(header_img, width=target_w, height=target_h)
			story.append(im)
			story.append(Spacer(1, 12))
		except Exception:
			pass

	# Title
	title_style = ParagraphStyle(
		name="TitleLarge",
		parent=styles["Title"],
		fontSize=24,
		leading=28,
		textColor=colors.HexColor("#0A2540"),
		spaceAfter=12,
	)
	title = Paragraph("Interview Evaluation Report", title_style)
	story.append(title)

	# Metadata
	meta_rows = [
		[Paragraph("Session ID", styles["MetaLabel"]), _para(meta.get("session_id"), styles["CellText"])],
		[Paragraph("Candidate", styles["MetaLabel"]), _para(meta.get("candidate_name"), styles["CellText"])],
		[Paragraph("Role", styles["MetaLabel"]), _para(meta.get("role"), styles["CellText"])],
		[
			Paragraph("Report Generated", styles["MetaLabel"]),
			_para(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), styles["CellText"]),
		],
		[
			Paragraph("Report Source", styles["MetaLabel"]),
			_para("Database interview_report", styles["CellText"]),
		],
	]
	t = Table(meta_rows, colWidths=[130, 380])
	t.setStyle(
		TableStyle(
			[
				("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
				("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#E0E6ED")),
				("ALIGN", (0, 0), (-1, -1), "LEFT"),
				("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
				("FONTSIZE", (0, 0), (-1, -1), 10),
				("BOTTOMPADDING", (0, 0), (-1, -1), 6),
				("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E6ED")),
			]
		)
	)
	story.append(Spacer(1, 6))
	story.append(t)
	story.append(Spacer(1, 18))

	# Summary section (if present)
	summary = report.get("evaluation_summary") or report.get("summary") or {}
	if isinstance(summary, dict) and summary:
		story.append(Paragraph("Summary", styles["SectionHeading"]))
		rows = []
		for key in ["overall_score", "overall_rating", "verdict", "confidence", "notes"]:
			if key in summary:
				rows.append([key.replace("_", " ").title(), _as_text(summary[key])])
		if rows:
			# Wrap all values to avoid overflow
			wrapped_rows = [[_para(k, styles["CellText"]), _para(v, styles["CellText"])] for k, v in rows]
			st = Table(wrapped_rows, colWidths=[180, 330])
			st.setStyle(
				TableStyle(
					[
						("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
						("FONTSIZE", (0, 0), (-1, -1), 11),
						("BOTTOMPADDING", (0, 0), (-1, -1), 6),
						("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
						("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#E0E6ED")),
						("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E6ED")),
						("WORDWRAP", (0, 0), (-1, -1), 'CJK'),
						("VALIGN", (0, 0), (-1, -1), 'TOP'),
						("LEFTPADDING", (0, 0), (-1, -1), 6),
						("RIGHTPADDING", (0, 0), (-1, -1), 6),
					]
				)
			)
			story.append(st)
			story.append(Spacer(1, 12))

	# Detailed scores (if present)
	detailed = report.get("detailed_scores") or {}
	if isinstance(detailed, dict) and detailed:
		story.append(Paragraph("Detailed Scores", styles["SectionHeading"]))
		for section, info in detailed.items():
			if isinstance(info, dict):
				title = section.replace("_", " ").title()
				story.append(Paragraph(title, styles["Heading3"]))
				rows = []
				if "overall_score" in info:
					rows.append(["Score", _as_text(info.get("overall_score"))])
				# Add any key:value pairs that look like metrics
				for k, v in info.items():
					if k == "overall_score":
						continue
					if isinstance(v, (str, int, float)):
						rows.append([k.replace("_", " ").title(), _as_text(v)])
				if rows:
					# Wrap section rows for safe word-wrapping
					wrapped = [[_para(a, styles["CellText"]), _para(b, styles["CellText"])] for a, b in rows]
					dt = Table(wrapped, colWidths=[180, 330])
					dt.setStyle(
						TableStyle(
							[
								("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
								("FONTSIZE", (0, 0), (-1, -1), 10),
								("BOTTOMPADDING", (0, 0), (-1, -1), 4),
								("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
								("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#E0E6ED")),
								("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E6ED")),
								("WORDWRAP", (0, 0), (-1, -1), 'CJK'),
								("VALIGN", (0, 0), (-1, -1), 'TOP'),
								("LEFTPADDING", (0, 0), (-1, -1), 6),
								("RIGHTPADDING", (0, 0), (-1, -1), 6),
							]
						)
					)
					story.append(dt)
					story.append(Spacer(1, 8))

				# Bulleted insights if present
				for key in ["strengths", "areas_for_improvement", "notes", "highlights"]:
					val = info.get(key)
					if isinstance(val, list) and val:
						story.append(Paragraph(key.replace("_", " ").title(), styles["Heading4"]))
						for item in val:
							bullet = Paragraph(f"• { _as_text(item) }", styles["BodyTextLarge"]) 
							story.append(bullet)
						story.append(Spacer(1, 6))

	# Recommendations (if present)
	recs = report.get("recommendations")
	if isinstance(recs, list) and recs:
		story.append(Paragraph("Recommendations", styles["SectionHeading"]))
		for r in recs:
			story.append(Paragraph(f"• { _as_text(r) }", styles["BodyTextLarge"]))
		story.append(Spacer(1, 12))

	# Transcript summary (optional)
	ts = report.get("transcript_summary")
	if ts:
		story.append(Paragraph("Transcript Summary", styles["SectionHeading"]))
		story.append(Paragraph(_as_text(ts), styles["BodyTextLarge"]))
		story.append(Spacer(1, 12))

	# Raw JSON fallback (optional)
	if not summary and not detailed and not recs and not ts:
		story.append(Paragraph("Report (Raw)", styles["SectionHeading"]))
		story.append(Paragraph(_as_text(report), styles["BodyTextLarge"]))
		story.append(Spacer(1, 12))

	return story


def generate_pdf_bytes(report: Dict[str, Any], meta: Dict[str, Any]) -> bytes:
	"""Generate a professional-looking PDF as bytes from the report JSON."""
	buffer = io.BytesIO()
	doc = SimpleDocTemplate(
		buffer,
		pagesize=LETTER,
		leftMargin=48,
		rightMargin=48,
		topMargin=48,
		bottomMargin=48,
		title="Interview Evaluation Report",
	)
	story = _build_pdf_story(report, meta)
	# Footer with page numbers
	def _on_page(canvas, d):
		canvas.saveState()
		canvas.setFont("Helvetica", 9)
		canvas.setFillColor(colors.HexColor("#6B7280"))
		page_txt = f"Page {d.page}"
		canvas.drawRightString(LETTER[0] - 48, 36, page_txt)
		canvas.restoreState()

	doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
	return buffer.getvalue()


def _ensure_bucket(client: Client, bucket: str, public: bool = True) -> None:
	"""Ensure the bucket exists; create it if missing (requires service role)."""
	try:
		resp = client.storage.list_buckets()
		# Normalize response to list of dicts
		if isinstance(resp, dict):
			bucket_list = resp.get("data") or resp.get("buckets") or []
		else:
			bucket_list = resp or []
		existing = any(((b or {}).get("name") == bucket) for b in bucket_list)
		if not existing:
			# Try multiple call signatures for broad SDK compatibility
			created = None
			try:
				created = client.storage.create_bucket(bucket, {"public": public})
			except Exception:
				try:
					created = client.storage.create_bucket(bucket_id=bucket, public=public)
				except Exception:
					try:
						created = client.storage.create_bucket({"id": bucket, "name": bucket, "public": public})
					except Exception:
						pass
			# Optionally verify after create
			return
	except Exception:
		# Best effort; continue even if ensure fails
		return


def upload_pdf_to_bucket(
	client: Client,
	bucket: str,
	dest_path: str,
	pdf_bytes: bytes,
) -> Dict[str, Any]:
	"""Upload PDF bytes to a Supabase Storage bucket and return upload info + public URL."""
	# Ensure path is normalized and unique-ish
	if not dest_path.lower().endswith(".pdf"):
		dest_path = dest_path + ".pdf"
	# Remove leading slashes and accidental bucket duplication in path
	dest_path = dest_path.lstrip("/")
	if dest_path.startswith(f"{bucket}/"):
		dest_path = dest_path[len(bucket) + 1:]

	# Ensure bucket exists (best effort)
	_ensure_bucket(client, bucket, public=True)

	# Upload (upsert allowed so reruns overwrite) using proper kwargs, with one retry on missing bucket
	def _do_upload():
		return client.storage.from_(bucket).upload(
			dest_path,
			pdf_bytes,
			{
				"contentType": "application/pdf",
				"cacheControl": "3600",
				"upsert": "true",
			},
		)
	try:
		res = _do_upload()
	except Exception as e:
		if "Bucket not found" in str(e):
			_ensure_bucket(client, bucket, public=True)
			res = _do_upload()
		else:
			raise
	# Get a public URL (requires bucket to be public)
	public_url = client.storage.from_(bucket).get_public_url(dest_path)
	# Normalize return to a string if SDK returns dict
	if isinstance(public_url, dict):
		public_url = (
			public_url.get("publicUrl")
			or public_url.get("public_url")
			or (public_url.get("data") or {}).get("publicUrl")
			or (public_url.get("data") or {}).get("public_url")
		)

	# Also create a signed URL for reliable access if bucket is private
	signed_url = None
	try:
		signed = client.storage.from_(bucket).create_signed_url(dest_path, 60 * 60 * 24 * 30)
		# Normalize different shapes
		if isinstance(signed, dict):
			# supabase-py often returns {'data': {'signedUrl': '...'} }
			signed_url = (
				(signed.get("data") or {}).get("signedUrl")
				or (signed.get("data") or {}).get("signedURL")
				or signed.get("signedUrl")
				or signed.get("signedURL")
			)
	except Exception:
		pass

	# Normalize URLs to absolute
	base_url = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
	def _absolutize(url: Optional[str]) -> Optional[str]:
		if not url:
			return url
		if url.startswith("http://") or url.startswith("https://"):
			return url
		if base_url:
			# Handle paths like '/storage/v1/object/...' or 'storage/v1/object/...'
			path = url
			if path.startswith("/"):
				return f"{base_url}{path}"
			return f"{base_url}/{path}"
		return url

	public_url = _absolutize(public_url)
	signed_url = _absolutize(signed_url)

	return {"upload": res, "public_url": public_url, "signed_url": signed_url}


def _update_pdf_url(client: Client, session_id: str, public_url: Optional[str]) -> None:
	"""Update the interview row with the generated PDF's public URL.

	Attempts updates against `interview_sessions` first, then `interview_session`,
	matching by `session_id` then by `id`.
	"""
	if not public_url:
		return
	payload = {"interview_report_pdf_url": public_url}
	for table_name in ["interview_sessions", "interview_session"]:
		try:
			resp = client.table(table_name).update(payload).eq("session_id", session_id).execute()
			if getattr(resp, "data", None):
				return
		except Exception:
			pass
		try:
			resp = client.table(table_name).update(payload).eq("id", session_id).execute()
			if getattr(resp, "data", None):
				return
		except Exception:
			pass
	# If all attempts fail, do not raise; keep operation best-effort
	return


def _get_existing_report_and_meta(client: Client, session_id: str) -> Dict[str, Any]:
	"""Fetch the existing interview report JSON and metadata from `interview_sessions.interview_report` with fallbacks."""
	# Primary: interview_sessions.interview_report
	try:
		resp = client.table("interview_sessions").select("*").eq("session_id", session_id).limit(1).execute()
		data = getattr(resp, "data", None) or []
		if not data:
			resp = client.table("interview_sessions").select("*").eq("id", session_id).limit(1).execute()
			data = getattr(resp, "data", None) or []
		if data:
			row = data[0]
			report = row.get("interview_report") or row.get("Interview_report")
			if report:
				meta = {
					"session_id": row.get("session_id") or row.get("id") or session_id,
					"candidate_name": row.get("candidate_name"),
					"role": row.get("role"),
					"created_at": row.get("created_at"),
				}
				return {"report": report, "meta": meta}
	except Exception:
		pass

	# Fallback: legacy table name interview_session
	try:
		resp = client.table("interview_session").select("*").eq("session_id", session_id).limit(1).execute()
		data = getattr(resp, "data", None) or []
		if not data:
			resp = client.table("interview_session").select("*").eq("id", session_id).limit(1).execute()
			data = getattr(resp, "data", None) or []
		if data:
			row = data[0]
			report = row.get("interview_report") or row.get("Interview_report")
			if report:
				meta = {
					"session_id": row.get("session_id") or row.get("id") or session_id,
					"candidate_name": row.get("candidate_name"),
					"role": row.get("role"),
					"created_at": row.get("created_at"),
				}
				return {"report": report, "meta": meta}
	except Exception:
		pass

	raise ValueError(f"No existing interview_report found for session {session_id}.")


def generate_and_upload_report(
	session_id: str,
	bucket: str = "interview-reports-gemini",
	destination_prefix: str = "interview-reports-gemini",
) -> Dict[str, Any]:
	"""End-to-end: fetch existing report JSON from DB, build PDF, and upload to storage.

	Returns a dict with keys: session_id, bucket, path, public_url, report_data
	"""
	_load_env()
	client = _init_supabase()

	# Fetch existing report and metadata
	retrieved = _get_existing_report_and_meta(client, session_id)
	report = retrieved["report"]
	# Allow for stringified JSON stored in the column
	if isinstance(report, str):
		try:
			report = json.loads(report)
		except Exception:
			pass
	# Ensure the report is a dict to avoid rendering errors
	if not isinstance(report, dict):
		report = {
			"evaluation_summary": {
				"overall_score": "N/A",
				"overall_rating": "N/A",
				"verdict": "Existing report is not a JSON object",
				"confidence": "Low",
				"notes": "The value in Interview_report could not be parsed into a JSON object. Including raw content below."
			},
			"raw_report": report
		}
	meta = retrieved["meta"]

	# Generate PDF directly from existing report
	pdf_bytes = generate_pdf_bytes(report, meta)

	# Store under a cleaner path: interview-reports-gemini/<session_id>/interview_report.pdf
	filename = "interview_report_gemini.pdf"
	dest_path = f"{destination_prefix}/{session_id}/{filename}"
	upload_info = upload_pdf_to_bucket(client, bucket, dest_path, pdf_bytes)

	# Persist a usable URL back to the interview row (prefer signed when present)
	usable_url = upload_info.get("signed_url") or upload_info.get("public_url")
	_update_pdf_url(client, session_id, usable_url)

	return {
		"session_id": session_id,
		"bucket": bucket,
		"path": dest_path,
		"public_url": upload_info.get("signed_url") or upload_info.get("public_url"),
		"report_data": report,
		"report_source": "existing_column"
	}

def _main_cli():
	import argparse

	parser = argparse.ArgumentParser(
		description="Build a PDF from an existing Interview_report in the database and upload it to Storage (no re-analysis)"
	)
	parser.add_argument("--session-id", "--session", required=True, help="Interview session UUID", dest="session_id")
	parser.add_argument(
		"--bucket",
		default="interview-report-pdf",
		help="Supabase Storage bucket to upload into",
	)
	parser.add_argument(
		"--prefix",
		default="interview-reports-gemini",
		help="Destination prefix/path in the bucket",
	)
	parser.add_argument(
		"--analysis-only",
		action="store_true",
		help=argparse.SUPPRESS
	)
	args = parser.parse_args()

	# Always build PDF from existing Interview_report
	result = generate_and_upload_report(
			session_id=args.session_id, 
			bucket=args.bucket, 
			destination_prefix=args.prefix
		)
	
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	_main_cli()
