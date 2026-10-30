"""
recommendations.py - Cautious, rule-based marketing recommendations and experimentation strategies.
"""

from __future__ import annotations

from typing import Any, Dict, List

RECOMMENDATION_DISCLAIMER = """
**Operational Disclaimer**:
These recommendations represent exploratory, hypothesis-driven marketing interventions based on historical
transaction clustering. Because unsupervised clustering does not capture customer sentiment, purchasing intent,
or competitive alternatives, teams should validate all actions via controlled A/B testing rather than deploying
blanket automated treatments.
""".strip()


def generate_cluster_recommendations(
    segment_meta: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Generate evidence-based, cautious marketing strategies and validation hypotheses for each segment.

    Parameters:
        segment_meta: Segment metadata dictionary from assign_evidence_based_names.

    Returns:
        Dict[int, Dict[str, Any]]: Segment ID mapping to strategic recommendations,
                                   retention tactics, risks, and A/B test hypotheses.
    """
    recommendations = {}

    for cid, meta in segment_meta.items():
        tag = meta["tag"]
        name = meta["name"]
        evidence = meta["evidence"]

        if tag == "Champions":
            rec = {
                "segment_name": name,
                "priority_level": "Highest (Value Preservation)",
                "strategic_goal": "Maintain exceptional engagement and maximize lifetime value without margin erosion.",
                "actionable_tactics": [
                    "Provide white-glove customer support and expedited fulfillment channels.",
                    "Offer exclusive early access to new product lines and seasonal collections.",
                    "Implement a tier-based loyalty reward program focusing on recognition over pure discounts.",
                    "Seek qualitative feedback or advisory panel participation to strengthen brand advocacy.",
                ],
                "pricing_strategy": "Avoid heavy discount coupons which unnecessarily dilute healthy margins.",
                "ab_test_hypothesis": (
                    "Hypothesis: Exclusive early access invitations will sustain repeat purchase rates equal to or "
                    "higher than cash discount promos, preserving gross margin by >5%."
                ),
                "risk_warning": "Over-messaging this sensitive group with aggressive sales promotions may induce banner fatigue.",
            }

        elif tag == "Core Active":
            rec = {
                "segment_name": name,
                "priority_level": "High (Basket Expansion)",
                "strategic_goal": "Increase average order value (AOV) while maintaining strong repeat frequency.",
                "actionable_tactics": [
                    "Introduce minimum spend thresholds for free shipping or volume discounts (e.g. 'Spend $50, Get Free Shipping').",
                    "Offer curated product bundles and complementary add-ons at checkout based on affinity.",
                    "Encourage multi-pack purchases with tiered unit pricing.",
                ],
                "pricing_strategy": "Incentivize basket size rather than single-item price slashing.",
                "ab_test_hypothesis": (
                    "Hypothesis: Offering free shipping on orders 20% above median basket size will increase AOV "
                    "by at least 15% without reducing monthly order frequency."
                ),
                "risk_warning": "Setting minimum order thresholds too high could deter habitual low-friction buyers.",
            }

        elif tag == "New Shoppers":
            rec = {
                "segment_name": name,
                "priority_level": "Medium-High (Activation)",
                "strategic_goal": "Encourage second purchase to transition new buyers into recurring customers.",
                "actionable_tactics": [
                    "Trigger an educational welcome email series showcasing catalog depth and customer reviews.",
                    "Send a time-limited second-purchase incentive within 14-21 days of initial order.",
                    "Collect post-purchase delivery feedback to immediately resolve any first-order friction.",
                ],
                "pricing_strategy": "Use modest, time-bound welcome incentives to encourage rapid repurchase.",
                "ab_test_hypothesis": (
                    "Hypothesis: A 10% coupon valid for 14 days will increase 30-day repurchase conversion by 25% "
                    "compared to a standard product recommendation email."
                ),
                "risk_warning": "Do not treat all first-time buyers as committed customers until repeat behavior is verified.",
            }

        elif tag == "At-Risk":
            rec = {
                "segment_name": name,
                "priority_level": "High (Revenue Recovery)",
                "strategic_goal": "Identify causes of dormancy and reactivate historically high-spending accounts.",
                "actionable_tactics": [
                    "Deploy targeted 'We Miss You' win-back campaigns featuring updated products or personalized offers.",
                    "Conduct structured outreach or survey to investigate whether service/quality issues caused defection.",
                    "Offer compelling incentives (e.g. substantial percentage discount or store credit) to prompt re-engagement.",
                ],
                "pricing_strategy": "Aggressive, high-impact incentives are economically justified given high historical customer spend.",
                "ab_test_hypothesis": (
                    "Hypothesis: Win-back messaging featuring high-margin new arrivals will yield a higher return "
                    "on marketing spend than generic site-wide sale notifications."
                ),
                "risk_warning": "Some accounts may have churned permanently or switched suppliers; cap outreach attempts at 2-3 to avoid brand annoyance.",
            }

        elif tag == "Slipping":
            rec = {
                "segment_name": name,
                "priority_level": "Medium (Re-Engagement)",
                "strategic_goal": "Re-awaken past habituated buyers before they transition into complete dormancy.",
                "actionable_tactics": [
                    "Send reminder emails highlighting replenishment of previously purchased staple items.",
                    "Share personalized catalog updates based on past browsing or order categories.",
                ],
                "pricing_strategy": "Moderate discount or limited-time free gift with purchase.",
                "ab_test_hypothesis": (
                    "Hypothesis: Automated re-order reminders based on estimated product lifecycle will achieve 10% reactivation."
                ),
                "risk_warning": "Lack of engagement might signal seasonal or one-off holiday shopping.",
            }

        else:  # Hibernating / Infrequent
            rec = {
                "segment_name": name,
                "priority_level": "Low (Cost Conservation)",
                "strategic_goal": "Minimize operational marketing costs while keeping low-cost re-engagement channels open.",
                "actionable_tactics": [
                    "Exclude from high-cost direct mail or paid retargeting advertising campaigns.",
                    "Include in quarterly automated newsletter digest or major clearance liquidation campaigns.",
                    "Purge completely unengaged accounts periodically to maintain email list deliverability health.",
                ],
                "pricing_strategy": "Do not invest significant marketing budget; rely on broad clearance promotions.",
                "ab_test_hypothesis": (
                    "Hypothesis: Ceasing paid retargeting on this cluster will reduce ad spend by 15% with no statistically significant drop in revenue."
                ),
                "risk_warning": "Spending expensive marketing dollars on dormant, low-value leads reduces overall campaign ROI.",
            }

        recommendations[cid] = rec

    return recommendations
