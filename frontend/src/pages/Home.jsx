
import React, { useState } from 'react';
import { 
  Brain, 
  Users, 
  GraduationCap, 
  FileText, 
  BarChart3, 
  Calendar, 
  MessageSquare, 
  Settings, 
  Bell, 
  Search, 
  AlertCircle, 
  TrendingUp, 
  CheckCircle, 
  Clock,
  Activity,
  Target,
  Award,
  Eye
} from 'lucide-react';
import './Home.css';
import Header from '../components/Header';

const Home = () => {
  const [activeSection, setActiveSection] = useState('dashboard');

   <Header></Header>
  const dashboardCards = [
    {
      title: 'Screening Assessment',
      icon: Brain,
      description: 'AI-powered handwriting analysis and dysgraphia screening tools for early detection',
      color: 'linear-gradient(135deg, #6366f1, #4f46e5)',
      stats: { assessments: 156, pending: 12, completed: 144 }
    },
    {
      title: 'Student Analytics',
      icon: BarChart3,
      description: 'Track writing progress, identify patterns, and monitor improvement over time',
      color: 'linear-gradient(135deg, #0891b2, #0e7490)',
      stats: { students: 89, flagged: 23, improving: 66 }
    },
    {
      title: 'Teacher Dashboard',
      icon: GraduationCap,
      description: 'Manage classes, view screening results, and create personalized intervention plans',
      color: 'linear-gradient(135deg, #10b981, #059669)',
      stats: { teachers: 15, classes: 42, reports: 78 }
    },
    {
      title: 'Parent Portal',
      icon: Users,
      description: 'Access your child\'s screening results, progress reports, and support resources',
      color: 'linear-gradient(135deg, #f59e0b, #d97706)',
      stats: { parents: 145, notifications: 8, meetings: 5 }
    }
  ];

  const quickActions = [
    { 
      icon: FileText, 
      label: 'New Assessment', 
      action: () => console.log('New Assessment'),
      description: 'Start screening'
    },
    { 
      icon: BarChart3, 
      label: 'View Reports', 
      action: () => console.log('View Reports'),
      description: 'Analysis results'
    },
    { 
      icon: Calendar, 
      label: 'Schedule Screening', 
      action: () => console.log('Schedule Screening'),
      description: 'Book appointment'
    },
    { 
      icon: MessageSquare, 
      label: 'Contact Support', 
      action: () => console.log('Contact Support'),
      description: 'Get help'
    }
  ];

  const recentActivity = [
    { 
      type: 'screening', 
      text: 'Sarah M. completed handwriting assessment - Early indicators detected', 
      time: '15 minutes ago', 
      color: '#6366f1', 
      priority: 'high' 
    },
    { 
      type: 'alert', 
      text: 'Alex K. flagged for dysgraphia screening - Requires immediate attention', 
      time: '1 hour ago', 
      color: '#ef4444', 
      priority: 'urgent' 
    },
    { 
      type: 'progress', 
      text: 'Emma R. showing 25% improvement in writing speed and accuracy', 
      time: '2 hours ago', 
      color: '#10b981', 
      priority: 'medium' 
    },
    { 
      type: 'report', 
      text: 'Monthly dysgraphia screening report generated for Grade 3', 
      time: '3 hours ago', 
      color: '#0891b2', 
      priority: 'low' 
    }
  ];

  
  const getPriorityClass = (priority) => {
    switch(priority) {
      case 'urgent': return 'priority-urgent';
      case 'high': return 'priority-high';
      case 'medium': return 'priority-medium';
      case 'low': return 'priority-low';
      default: return 'priority-low';
    }
  };

  return (
    <div className="container">
     
      {/* Main Content */}
      <main className="main">
        {/* Welcome Section */}
        <div className="welcome">
          <h2 className="welcome-title">Dysgraphia Screening Dashboard</h2>
          <p className="welcome-subtitle">
            Early detection and intervention platform for identifying dysgraphia in students
          </p>
        </div>

       

        {/* Dashboard Cards */}
        <div className="card-grid">
          {dashboardCards.map((card, index) => {
            const IconComponent = card.icon;
            return (
              <div
                key={index}
                className="card"
                style={{ background: card.color }}
                onClick={() => setActiveSection(card.title.toLowerCase().replace(' ', '-'))}
              >
                <div className="card-header">
                  <IconComponent size={32} />
                  <div className="card-stats">
                    <div className="card-stats-label">Total</div>
                    <div className="card-stats-value">
                      {Object.values(card.stats)[0]}
                    </div>
                  </div>
                </div>
                <h3 className="card-title">{card.title}</h3>
                <p className="card-description">{card.description}</p>
                <div className="card-footer">
                  {Object.entries(card.stats).map(([key, value], idx) => (
                    <span key={idx}>
                      {key}: {value}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Quick Actions and Recent Activity */}
        <div className="two-column-grid">
          {/* Quick Actions */}
          <div className="white-card">
            <h3 className="section-title">Quick Actions</h3>
            <div className="quick-actions-grid">
              {quickActions.map((action, index) => {
                const IconComponent = action.icon;
                return (
                  <button
                    key={index}
                    className="quick-action-button"
                    onClick={action.action}
                  >
                    <IconComponent size={20} color="#6366f1" />
                    <div>
                      <div style={{ fontWeight: '600' }}>{action.label}</div>
                      <div style={{ fontSize: '12px', color: '#64748b' }}>
                        {action.description}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="white-card">
            <h3 className="section-title">Recent Activity</h3>
            <div className="activity-list">
              {recentActivity.map((activity, index) => (
                <div key={index} className="activity-item">
                  <div 
                    className="activity-dot" 
                    style={{ backgroundColor: activity.color }}
                  ></div>
                  <div className="activity-content">
                    <p className="activity-text">
                      {activity.text}
                      <span className={`priority-badge ${getPriorityClass(activity.priority)}`}>
                        {activity.priority}
                      </span>
                    </p>
                    <p className="activity-time">
                      <Clock size={12} style={{ marginRight: '4px' }} />
                      {activity.time}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

      
      </main>
    </div>
  );
};

export default Home;