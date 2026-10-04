import Link from 'next/link';

const journeyStages = [
  {
    number: '01',
    moment: 'When your records are scattered',
    engine: 'LANGUAGE AI',
    title: 'Bring the story together',
    body: 'Language AI reads scans, pathology, treatments, genomics, and labs and connects the important details across reports.',
    value: 'One clear cancer history instead of disconnected records.',
  },
  {
    number: '02',
    moment: 'When you need the current picture',
    engine: 'CONNECTED CANCER HISTORY',
    title: 'Understand where things stand',
    body: 'OncoTwin connects treatments, biomarkers, disease sites, scans, and missing information into an evolving view of your cancer.',
    value: 'A clearer picture of what is known now and what has changed.',
  },
  {
    number: '03',
    moment: 'When a new scan arrives',
    engine: 'CANCER PROGRESSION MODEL',
    title: 'See what a new scan changes',
    body: 'A machine-learning model specially trained to predict cancer progression uses your verified history and each new scan to update its view of your trajectory.',
    value: 'See not just what the scan reported, but how it changes the broader picture over time.',
  },
  {
    number: '04',
    moment: 'Before your next appointment',
    engine: 'AI EXPLANATION',
    title: 'Know what to focus on next',
    body: 'AI explains what changed, what remains uncertain, and what may be worth discussing with your care team.',
    value: 'Walk into the next conversation with a clearer understanding of what matters.',
  },
];

export default function LandingPage() {
  return (
    <main className="landing landing-v3">
      <nav className="landing-nav landing-nav-v3" aria-label="Main navigation">
        <Link href="/" className="brand" aria-label="OncoTwin home">
          <span className="brand-mark">O</span>
          <span>OncoTwin</span>
        </Link>
      </nav>

      <section className="landing-hero-v3">
        <div className="landing-hero-copy-v3">
          <div className="eyebrow">AI FOR YOUR CANCER JOURNEY</div>
          <h1>Understand how your cancer is changing—not just what each report says.</h1>
          <p className="landing-hero-lede-v3">
            OncoTwin uses AI to read and connect your scans, pathology, treatments, genomics, and labs into one evolving cancer history. A machine-learning model specially trained to predict cancer progression then uses that history and each new scan to update its view of your trajectory over time.
          </p>
          <p className="landing-hero-impact-v3">
            See what changed, how the bigger picture shifted, and what to focus on next.
          </p>
          <div className="hero-actions">
            <Link className="primary-button" href="/app">Open my cancer story</Link>
            <a className="secondary-button" href="#journey">See how it helps</a>
          </div>
        </div>
      </section>

      <section className="journey-section journey-section-v3" id="journey">
        <div className="journey-section-head-v3">
          <div className="eyebrow">ALONG THE CANCER JOURNEY</div>
          <h2>How OncoTwin helps as things change</h2>
          <p>Different kinds of AI help at different moments in your cancer journey.</p>
        </div>

        <div className="journey-grid-v3">
          {journeyStages.map((stage, index) => (
            <article className="journey-card-v3" key={stage.number}>
              <div className="journey-card-top-v3">
                <span className="journey-number-v3">{stage.number}</span>
                <span className="journey-engine-v3">{stage.engine}</span>
              </div>
              <p className="journey-moment-v3">{stage.moment}</p>
              <h3>{stage.title}</h3>
              <p className="journey-body-v3">{stage.body}</p>
              <div className="journey-value-v3">
                <span>WHAT YOU GET</span>
                <strong>{stage.value}</strong>
              </div>
              {index < journeyStages.length - 1 ? <span className="journey-arrow-v3" aria-hidden="true">→</span> : null}
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
