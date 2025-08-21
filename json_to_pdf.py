import os
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai

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
os.environ['SUPABASE_URL'] = 'https://ibnsjeoemngngkqnnjdz.supabase.co'
os.environ['SUPABASE_SERVICE_ROLE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlibnNqZW9lbW5nbmdrcW5uamR6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MzcwOTgxMSwiZXhwIjoyMDY5Mjg1ODExfQ.9Qr2srBzKeVLkZcq1ZMv-B2-_mj71QyDTzdedgxSCSs'
os.environ['SUPABASE_ANON_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlibnNqZW9lbW5nbmdrcW5uamR6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3MDk4MTEsImV4cCI6MjA2OTI4NTgxMX0.iR8d0XxR-UOPPrK74IIV6Z7gVPP2rHS2b1ZCKwGOSqQ'


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


def _init_gemini() -> genai.GenerativeModel:
	"""Initialize and return a Gemini API client using env vars.

	Required env var:
	- GEMINI_API_KEY
	"""
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("Missing GEMINI_API_KEY env var")
	
	genai.configure(api_key=api_key)
	return genai.GenerativeModel('gemini-1.5-flash')


def _get_interview_data(
	client: Client, session_id: str
) -> Dict[str, Any]:
	"""Fetch interview data for a given session_id.

	This fetches the raw interview data (transcript, questions, etc.) that will be
	processed by Gemini to generate the report.
	"""
	# Select all columns to avoid errors when specific columns don't exist
	resp = (
		client.table("interview_sessions").select("*").eq("session_id", session_id).limit(1).execute()
	)

	data = getattr(resp, "data", None) or []
	if not data:
		# Some schemas may use `id` instead of `session_id`
		resp = (
			client.table("interview_sessions").select("*").eq("id", session_id).limit(1).execute()
		)
		data = getattr(resp, "data", None) or []

	if not data:
		raise ValueError(f"No interview session found for session_id/id: {session_id}")

	row = data[0]
	
	# Extract relevant data for analysis
	transcript = row.get("transcript") or row.get("interview_transcript")
	questions = row.get("questions") or row.get("interview_questions")
	
	# Metadata
	meta = {
		"session_id": row.get("session_id") or row.get("id") or session_id,
		"candidate_name": row.get("candidate_name"),
		"role": row.get("role"),
		"created_at": row.get("created_at"),
	}
	
	return {
		"transcript": transcript,
		"questions": questions,
		"meta": meta,
		"raw_data": row
	}


def _generate_report_with_gemini(model: genai.GenerativeModel, transcript: str, questions: str, role: str) -> Dict[str, Any]:
	"""Use Gemini API to analyze interview data and generate a structured report."""
	
	prompt = f"""
	You are an expert interview evaluator. Analyze the following interview transcript and questions for a {role} position.
	
	Interview Questions:
	{questions if questions else "No structured questions provided"}
	
	Interview Transcript:
	{transcript if transcript else "No transcript provided"}
	
	Please provide a comprehensive evaluation in the following JSON format:
	{{
		"evaluation_summary": {{
			"overall_score": "Score out of 10",
			"overall_rating": "Excellent/Good/Average/Below Average/Poor",
			"verdict": "Hire/Consider/Reject with reasoning",
			"confidence": "High/Medium/Low confidence in assessment",
			"notes": "Key observations and summary"
		}},
		"detailed_scores": {{
			"technical_skills": {{
				"overall_score": "Score out of 10",
				"strengths": ["List of technical strengths"],
				"areas_for_improvement": ["Areas needing development"],
				"notes": "Detailed technical assessment"
			}},
			"communication": {{
				"overall_score": "Score out of 10",
				"strengths": ["Communication strengths"],
				"areas_for_improvement": ["Communication areas for improvement"],
				"notes": "Communication skills assessment"
			}},
			"problem_solving": {{
				"overall_score": "Score out of 10",
				"strengths": ["Problem-solving strengths"],
				"areas_for_improvement": ["Problem-solving areas for improvement"],
				"notes": "Problem-solving assessment"
			}},
			"cultural_fit": {{
				"overall_score": "Score out of 10",
				"strengths": ["Cultural fit strengths"],
				"areas_for_improvement": ["Areas for cultural alignment"],
				"notes": "Cultural fit assessment"
			}}
		}},
		"recommendations": [
			"Specific recommendations for the candidate",
			"Suggestions for next steps",
			"Development areas if hired"
		],
		"transcript_summary": "Concise summary of the key points from the interview"
	}}
	
	Please ensure the response is valid JSON and provide thoughtful, detailed analysis based on the interview content.
	If the transcript or questions are missing or incomplete, please note this in your assessment and provide what analysis you can.
	"""
	
	try:
		response = model.generate_content(prompt)
		
		# Extract text from response
		if hasattr(response, 'text'):
			response_text = response.text
		elif hasattr(response, 'content') and hasattr(response.content, 'parts'):
			response_text = response.content.parts[0].text
		else:
			response_text = str(response)
		
		# Clean up response text - remove markdown code blocks if present
		response_text = response_text.strip()
		if response_text.startswith('```json'):
			response_text = response_text[7:]  # Remove ```json
		if response_text.startswith('```'):
			response_text = response_text[3:]   # Remove ```
		if response_text.endswith('```'):
			response_text = response_text[:-3]  # Remove closing ```
		response_text = response_text.strip()
		
		# Parse JSON response
		report = json.loads(response_text)
		return report
		
	except json.JSONDecodeError as e:
		print(f"Error parsing Gemini response as JSON: {e}")
		print(f"Raw response: {response_text[:500]}...")
		# Return a basic structure with the raw response
		return {
			"evaluation_summary": {
				"overall_score": "N/A",
				"overall_rating": "Unable to assess",
				"verdict": "Analysis failed - JSON parsing error",
				"confidence": "Low",
				"notes": f"Gemini response could not be parsed as JSON: {str(e)}"
			},
			"raw_gemini_response": response_text
		}
	except Exception as e:
		print(f"Error generating report with Gemini: {e}")
		return {
			"evaluation_summary": {
				"overall_score": "N/A",
				"overall_rating": "Unable to assess",
				"verdict": f"Analysis failed - {str(e)}",
				"confidence": "Low",
				"notes": f"Error occurred during Gemini analysis: {str(e)}"
			}
		}


def _update_interview_report(client: Client, session_id: str, report: Dict[str, Any]) -> None:
	"""Update the interview session with the generated report."""
	try:
		# Prepare base update data
		base_update = {
			"Interview_report": report
		}
		
		# Try to add optional columns if they exist
		optional_columns = {
			"report_generated_at": datetime.now(timezone.utc).isoformat(),
			"report_source": "gemini"
		}
		
		# Try updating with Interview_report column first (preferred)
		try:
			resp = client.table("interview_sessions").update({
				**base_update,
				**optional_columns
			}).eq("session_id", session_id).execute()
			
			if getattr(resp, "data", None):
				return  # Success!
		except Exception as e:
			# If optional columns fail, try with just the base update
			if "could not find" in str(e).lower() or "pgrst204" in str(e).lower():
				print(f"Optional columns not available, using base update only: {e}")
				try:
					resp = client.table("interview_sessions").update(base_update).eq("session_id", session_id).execute()
					if getattr(resp, "data", None):
						return  # Success!
				except Exception:
					pass
			else:
				print(f"Error with session_id lookup: {e}")
		
		# Try with id column instead
		try:
			resp = client.table("interview_sessions").update({
				**base_update,
				**optional_columns
			}).eq("id", session_id).execute()
			
			if getattr(resp, "data", None):
				return  # Success!
		except Exception as e:
			# If optional columns fail, try with just the base update
			if "could not find" in str(e).lower() or "pgrst204" in str(e).lower():
				try:
					resp = client.table("interview_sessions").update(base_update).eq("id", session_id).execute()
					if getattr(resp, "data", None):
						return  # Success!
				except Exception:
					pass
			else:
				print(f"Error with id lookup: {e}")
		
		# Try with interview_report column (legacy)
		legacy_update = {
			"interview_report": report
		}
		
		try:
			resp = client.table("interview_sessions").update({
				**legacy_update,
				**optional_columns
			}).eq("session_id", session_id).execute()
			
			if getattr(resp, "data", None):
				return  # Success!
		except Exception as e:
			# If optional columns fail, try with just the legacy update
			if "could not find" in str(e).lower() or "pgrst204" in str(e).lower():
				try:
					resp = client.table("interview_sessions").update(legacy_update).eq("session_id", session_id).execute()
					if getattr(resp, "data", None):
						return  # Success!
				except Exception:
					pass
		
		# Final attempt with id and legacy column
		try:
			resp = client.table("interview_sessions").update({
				**legacy_update,
				**optional_columns
			}).eq("id", session_id).execute()
			
			if getattr(resp, "data", None):
				return  # Success!
		except Exception as e:
			# If optional columns fail, try with just the legacy update
			if "could not find" in str(e).lower() or "pgrst204" in str(e).lower():
				try:
					resp = client.table("interview_sessions").update(legacy_update).eq("id", session_id).execute()
					if getattr(resp, "data", None):
						return  # Success!
				except Exception:
					pass
		
		# If we get here, all attempts failed
		raise Exception(f"Failed to update interview report for session {session_id}. Tried all column combinations.")
				
	except Exception as e:
		print(f"Error updating interview report in database: {e}")
		raise


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
	title = Paragraph("Interview Evaluation Report (Generated by Gemini AI)", title_style)
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
			Paragraph("Analysis Engine", styles["MetaLabel"]),
			_para("Google Gemini AI", styles["CellText"]),
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
		title="Interview Evaluation Report (Gemini AI)",
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
		buckets = client.storage.list_buckets()
		existing = any((b.get("name") == bucket) for b in (buckets or []))
		if not existing:
			client.storage.create_bucket(bucket, public=public)
	except Exception:
		# If the SDK version doesn't support list/create in this environment, ignore and proceed.
		pass


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

	# Ensure bucket exists (best effort)
	_ensure_bucket(client, bucket, public=True)

	# Upload (upsert allowed so reruns overwrite) using proper kwargs
	res = client.storage.from_(bucket).upload(
		dest_path,
		pdf_bytes,
		{
			"contentType": "application/pdf",
			"cacheControl": "3600",
			"upsert": "true",
		},
	)
	# Get a public URL (requires bucket to be public or signed URLs logic)
	public_url = client.storage.from_(bucket).get_public_url(dest_path)
	# Normalize return to a string if SDK returns dict
	if isinstance(public_url, dict):
		public_url = public_url.get("data") or public_url.get("publicUrl") or public_url.get("public_url")
	return {"upload": res, "public_url": public_url}


def generate_and_upload_report(
	session_id: str,
	bucket: str = "interview-report-pdf",
	destination_prefix: str = "interview-reports-gemini",
) -> Dict[str, Any]:
	"""End-to-end: fetch interview data from DB, analyze with Gemini, build PDF, and upload to storage.

	Returns a dict with keys: session_id, bucket, path, public_url, report_data
	"""
	_load_env()
	client = _init_supabase()
	model = _init_gemini()

	# Fetch interview data
	interview_data = _get_interview_data(client, session_id)
	
	# Check if evaluation report already exists
	existing_report = interview_data.get("Interview_report")
	if existing_report and isinstance(existing_report, dict) and existing_report.get("evaluation_summary"):
		print(f"Using existing evaluation report for session {session_id}")
		report = existing_report
	else:
		print(f"No existing evaluation found, generating new report with Gemini for session {session_id}")
		# Generate report using Gemini
		report = _generate_report_with_gemini(
			model, 
			interview_data.get("transcript", ""), 
			interview_data.get("questions", ""),
			interview_data["meta"].get("role", "Unknown Role")
		)
		
		# Update database with generated report
		_update_interview_report(client, session_id, report)
	
	# Generate PDF
	pdf_bytes = generate_pdf_bytes(report, interview_data["meta"])

	# Store under a cleaner path: interview-reports-gemini/<session_id>/interview_report.pdf
	filename = "interview_report_gemini.pdf"
	dest_path = f"{destination_prefix}/{session_id}/{filename}"
	upload_info = upload_pdf_to_bucket(client, bucket, dest_path, pdf_bytes)

	return {
		"session_id": session_id,
		"bucket": bucket,
		"path": dest_path,
		"public_url": upload_info.get("public_url"),
		"report_data": report,
		"analysis_engine": "gemini"
	}


def analyze_interview_only(session_id: str) -> Dict[str, Any]:
	"""Only analyze interview data with Gemini and update database (no PDF generation).
	
	This function is useful when you only want to generate the analysis
	without creating a PDF report.
	"""
	_load_env()
	client = _init_supabase()
	model = _init_gemini()

	# Fetch interview data
	interview_data = _get_interview_data(client, session_id)
	
	# Check if evaluation report already exists
	existing_report = interview_data.get("Interview_report")
	if existing_report and isinstance(existing_report, dict) and existing_report.get("evaluation_summary"):
		print(f"Using existing evaluation report for session {session_id}")
		report = existing_report
	else:
		print(f"No existing evaluation found, generating new report with Gemini for session {session_id}")
		# Generate report using Gemini
		report = _generate_report_with_gemini(
			model, 
			interview_data.get("transcript", ""), 
			interview_data.get("questions", ""),
			interview_data["meta"].get("role", "Unknown Role")
		)
		
		# Update database with generated report
		_update_interview_report(client, session_id, report)
	
	return {
		"session_id": session_id,
		"report_data": report,
		"analysis_engine": "gemini",
		"status": "analysis_complete"
	}


def _main_cli():
	import argparse

	parser = argparse.ArgumentParser(
		description="Generate an interview report using Gemini AI from Supabase data and upload PDF to Storage"
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
		help="Only perform analysis and update database (no PDF generation)"
	)
	args = parser.parse_args()

	if args.analysis_only:
		result = analyze_interview_only(session_id=args.session_id)
	else:
		result = generate_and_upload_report(
			session_id=args.session_id, 
			bucket=args.bucket, 
			destination_prefix=args.prefix
		)
	
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	_main_cli()
